from pathlib import Path
import json, os, sys, traceback
report={"python":sys.version,"home":str(Path.home()),"files":[]}
try:
    from f5_tts_th.tts import TTS
    report["tts_import"]="ok"
    model=TTS(model="v1")
    report["tts_init"]="ok"
    report["tts_type"]=str(type(model))
    report["tts_attrs"]={k:str(v)[:500] for k,v in vars(model).items() if isinstance(v,(str,int,float,bool,Path))}
except Exception as e:
    report["error"]=repr(e); report["traceback"]=traceback.format_exc()
for root in [Path.home()/".cache",Path.home()/"AppData"/"Local",Path.cwd()]:
    if not root.exists(): continue
    try:
        for p in root.rglob("*"):
            if p.is_file() and (p.suffix.lower() in {".pt",".pth",".safetensors",".yaml",".yml"} or p.name=="vocab.txt"):
                report["files"].append({"path":str(p),"size":p.stat().st_size})
    except Exception as e: report.setdefault("scan_errors",[]).append(str(e))
Path("model_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
if "error" in report: raise SystemExit(1)
