from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from f5_tts_th.tts import TTS


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build_work"
VENDOR = WORK / "F5-TTS-ONNX"
OUTPUT = ROOT / "onnx_models"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def find_thai_assets() -> tuple[Path, Path]:
    TTS(model="v1")
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    checkpoints = sorted(cache.glob("models--VIZINTZOR--F5-TTS-THAI/snapshots/*/model_1000000.pt"))
    vocabs = sorted(cache.glob("models--VIZINTZOR--F5-TTS-THAI/snapshots/*/vocab.txt"))
    if not checkpoints or not vocabs:
        raise FileNotFoundError("Thai checkpoint or vocab.txt was not downloaded")
    return checkpoints[-1], vocabs[-1]


def patch_pt_architecture(exporter: Path) -> None:
    source = exporter.read_text(encoding="utf-8")
    pattern = r"def get_checkpoint_architecture\(checkpoint_path: Path\) -> tuple\[int, int\]:.*?\n\ndef is_compatible_model_config\("
    replacement = '''def get_checkpoint_architecture(checkpoint_path: Path) -> tuple[int, int]:
    if checkpoint_path.suffix.casefold() == ".safetensors":
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            projection_keys = [key for key in keys if key.endswith("transformer.proj_out.weight")]
            if len(projection_keys) != 1:
                raise ValueError(f"Could not infer projection layer from {checkpoint_path}")
            hidden_size = checkpoint.get_slice(projection_keys[0]).get_shape()[1]
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict", checkpoint))
        keys = list(state.keys())
        projection_keys = [key for key in keys if key.endswith("transformer.proj_out.weight")]
        if len(projection_keys) != 1:
            raise ValueError(f"Could not infer projection layer from {checkpoint_path}")
        hidden_size = state[projection_keys[0]].shape[1]
    block_indices = {
        int(key.split("transformer.transformer_blocks.", 1)[1].split(".", 1)[0])
        for key in keys if "transformer.transformer_blocks." in key
    }
    if not block_indices:
        raise ValueError(f"Could not infer transformer depth from {checkpoint_path}")
    return int(hidden_size), max(block_indices) + 1


def is_compatible_model_config('''
    patched, count = re.subn(pattern, replacement, source, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Exporter patch expected one match, found {count}")
    exporter.write_text(patched, encoding="utf-8")


def main() -> None:
    WORK.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    if not VENDOR.exists():
        run("git", "clone", "--depth", "1", "https://github.com/DakeQQ/F5-TTS-ONNX.git", str(VENDOR))

    checkpoint, vocab = find_thai_assets()
    vocos = Path(snapshot_download(repo_id="charactr/vocos-mel-24khz"))
    exporter = VENDOR / "Export_ONNX" / "F5_TTS" / "Export_F5.py"
    patch_pt_architecture(exporter)

    f5_tts = importlib.import_module("f5_tts")
    configs = [p for root in f5_tts.__path__ for p in (Path(root) / "configs").glob("*.yaml")]
    v1_configs = [p for p in configs if "v1" in p.name.casefold() and "base" in p.name.casefold()]
    if len(v1_configs) != 1:
        raise RuntimeError(f"Expected one F5 v1 Base config, found: {v1_configs}")
    config = v1_configs[0]

    run(
        sys.executable, str(exporter),
        "--model_series", "v1",
        "--f5safetensor_path", str(checkpoint),
        "--vocab_path", str(vocab),
        "--omegacfg_path", str(config),
        "--vocosmodel_dir", str(vocos),
        "--preprocessmodel_path", str(OUTPUT / "F5_Preprocess.onnx"),
        "--transformermodel_path", str(OUTPUT / "F5_Transformer.onnx"),
        "--decodermodel_path", str(OUTPUT / "F5_Decode.onnx"),
        "--metadatamodel_path", str(OUTPUT / "F5_Metadata.onnx"),
        "--skip_inference",
    )
    (OUTPUT / "vocab.txt").write_bytes(vocab.read_bytes())
    required = ["F5_Preprocess.onnx", "F5_Transformer.onnx", "F5_Decode.onnx", "F5_Metadata.onnx", "vocab.txt"]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing exported files: {missing}")
    print("Thai ONNX export completed:", {p.name: p.stat().st_size for p in OUTPUT.iterdir()})


if __name__ == "__main__":
    main()
