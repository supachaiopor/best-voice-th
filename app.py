from __future__ import annotations
import json, os, runpy, shutil, subprocess, sys, threading, time, wave
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

ROOT=Path(getattr(sys,"_MEIPASS",Path(__file__).resolve().parent))
APP=Path(sys.executable).resolve().parent if getattr(sys,"frozen",False) else Path(__file__).resolve().parent
DATA=APP/"data"; VOICES=DATA/"voices"; OUTPUTS=DATA/"outputs"; MODELS=APP/"models"
for p in (DATA,VOICES,OUTPUTS): p.mkdir(parents=True,exist_ok=True)
PROFILE_FILE=VOICES/"profiles.json"

def engine_mode():
    if "--engine" not in sys.argv: return False
    i=sys.argv.index("--engine"); args=sys.argv[i+1:]
    script=ROOT/"vendor"/"Inference_F5_TTS_ONNX.py"
    sys.argv=[str(script)]+args
    runpy.run_path(str(script),run_name="__main__")
    return True

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
        self.title("BEST Voice TH — Offline Voice Studio")
        self.geometry("1220x760"); self.minsize(1050,680)
        self.profiles=self.load_profiles(); self.current=None; self.started=0
        self.grid_columnconfigure(1,weight=1); self.grid_rowconfigure(1,weight=1)
        self.header(); self.sidebar(); self.main(); self.refresh_profiles()
    def header(self):
        f=ctk.CTkFrame(self,corner_radius=0,height=82); f.grid(row=0,column=0,columnspan=2,sticky="nsew")
        ctk.CTkLabel(f,text="◉  BEST VOICE TH",font=("Tahoma",24,"bold")).pack(side="left",padx=25,pady=20)
        self.device=ctk.CTkLabel(f,text=self.provider(),text_color="#35deb4",font=("Tahoma",12,"bold")); self.device.pack(side="right",padx=25)
    def sidebar(self):
        f=ctk.CTkFrame(self,width=300,corner_radius=0); f.grid(row=1,column=0,sticky="nsew",padx=(14,7),pady=14); f.grid_propagate(False)
        ctk.CTkLabel(f,text="เสียงของฉัน",font=("Tahoma",18,"bold")).pack(anchor="w",padx=18,pady=(20,3))
        ctk.CTkLabel(f,text="เพิ่มเสียงต้นแบบ 3–10 วินาที",text_color="#91a5bc").pack(anchor="w",padx=18)
        self.list=tk.Listbox(f,bg="#0b1727",fg="white",selectbackground="#176077",bd=0,font=("Tahoma",11),height=10)
        self.list.pack(fill="x",padx=18,pady=15); self.list.bind("<<ListboxSelect>>",self.select)
        ctk.CTkButton(f,text="+ เพิ่มเสียง",command=self.add_voice,fg_color="#156079").pack(fill="x",padx=18)
        ctk.CTkButton(f,text="ลบเสียงที่เลือก",command=self.delete_voice,fg_color="#472632",hover_color="#663040").pack(fill="x",padx=18,pady=(8,0))
        ctk.CTkLabel(f,text="ข้อความในเสียงต้นแบบ",text_color="#91a5bc").pack(anchor="w",padx=18,pady=(22,5))
        self.reftext=ctk.CTkTextbox(f,height=135,font=("Tahoma",11)); self.reftext.pack(fill="x",padx=18)
        self.status=ctk.CTkLabel(f,text="พร้อมใช้งาน",text_color="#35deb4"); self.status.pack(side="bottom",anchor="w",padx=18,pady=18)
    def main(self):
        f=ctk.CTkFrame(self,corner_radius=0); f.grid(row=1,column=1,sticky="nsew",padx=(7,14),pady=14)
        f.grid_columnconfigure(0,weight=1); f.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(f,text="สร้างเสียง",font=("Tahoma",20,"bold")).grid(row=0,column=0,sticky="w",padx=22,pady=(20,8))
        self.text=ctk.CTkTextbox(f,font=("Tahoma",14)); self.text.grid(row=1,column=0,sticky="nsew",padx=22)
        controls=ctk.CTkFrame(f,fg_color="transparent"); controls.grid(row=2,column=0,sticky="ew",padx=22,pady=16)
        self.speed=ctk.DoubleVar(value=1.0); self.steps=ctk.IntVar(value=16); self.backend=ctk.StringVar(value="อัตโนมัติ")
        for title,var,vals in [("ประมวลผล",self.backend,["อัตโนมัติ","CPU","DirectML"]),("ความเร็ว",self.speed,["0.8","0.9","1.0","1.1","1.2"]),("คุณภาพ/NFE",self.steps,["8","16","24","32"])]:
            box=ctk.CTkFrame(controls,fg_color="transparent"); box.pack(side="left",fill="x",expand=True,padx=(0,10))
            ctk.CTkLabel(box,text=title,text_color="#91a5bc").pack(anchor="w")
            ctk.CTkOptionMenu(box,variable=var,values=vals).pack(fill="x",pady=(4,0))
        self.go=ctk.CTkButton(controls,text="✨ สร้างเสียง",font=("Tahoma",14,"bold"),height=52,command=self.generate,fg_color="#f59e0b",text_color="#111")
        self.go.pack(side="right",padx=(8,0))
        player=ctk.CTkFrame(f); player.grid(row=3,column=0,sticky="ew",padx=22,pady=(0,18))
        ctk.CTkButton(player,text="▶ เล่น",width=90,command=self.play).pack(side="left",padx=10,pady=10)
        self.outlabel=ctk.CTkLabel(player,text="ยังไม่มีไฟล์เสียง",text_color="#91a5bc"); self.outlabel.pack(side="left",fill="x",expand=True)
        ctk.CTkButton(player,text="Export WAV",width=120,command=self.export).pack(side="right",padx=10)
    def provider(self):
        try:
            import onnxruntime as ort
            ps=ort.get_available_providers()
            return "● DirectML" if "DmlExecutionProvider" in ps else "● CPU"
        except Exception:return "Engine ไม่พร้อม"
    def load_profiles(self):
        try:return json.loads(PROFILE_FILE.read_text("utf-8"))
        except Exception:return []
    def save_profiles(self):PROFILE_FILE.write_text(json.dumps(self.profiles,ensure_ascii=False,indent=2),"utf-8")
    def refresh_profiles(self):
        self.list.delete(0,"end")
        for p in self.profiles:self.list.insert("end",p["name"])
    def add_voice(self):
        audio=filedialog.askopenfilename(title="เลือกเสียงต้นแบบ",filetypes=[("Audio","*.wav *.mp3 *.flac *.m4a")])
        if not audio:return
        name=tk.simpledialog.askstring("ชื่อเสียง","ตั้งชื่อเสียงนี้:",parent=self)
        if not name:return
        dst=VOICES/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+Path(audio).name); shutil.copy2(audio,dst)
        self.profiles.append({"name":name.strip(),"audio":str(dst),"text":""}); self.save_profiles(); self.refresh_profiles()
    def select(self,_=None):
        s=self.list.curselection()
        if not s:return
        p=self.profiles[s[0]]; self.reftext.delete("1.0","end"); self.reftext.insert("1.0",p.get("text",""))
    def delete_voice(self):
        s=self.list.curselection()
        if not s:return
        self.profiles.pop(s[0]); self.save_profiles(); self.refresh_profiles()
    def generate(self):
        s=self.list.curselection(); text=self.text.get("1.0","end").strip(); ref=self.reftext.get("1.0","end").strip()
        if not s:return messagebox.showwarning("ยังไม่ได้เลือกเสียง","เพิ่มและเลือกเสียงต้นแบบก่อน")
        if not text or not ref:return messagebox.showwarning("ข้อมูลไม่ครบ","ใส่ข้อความต้นแบบและข้อความที่จะสร้าง")
        for n in ("F5_Metadata.onnx","F5_Preprocess.onnx","F5_Transformer.onnx","F5_Decode.onnx","vocab.txt"):
            if not (MODELS/n).exists():return messagebox.showerror("โมเดลไม่ครบ",f"ไม่พบ {n}")
        p=self.profiles[s[0]]; p["text"]=ref; self.save_profiles()
        out=OUTPUTS/("best_voice_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".wav")
        provider="DmlExecutionProvider" if self.backend.get()!="CPU" and "DmlExecutionProvider" in __import__("onnxruntime").get_available_providers() else "CPUExecutionProvider"
        args=["--engine","--onnx-folder",str(MODELS),"--vocab-path",str(MODELS/"vocab.txt"),"--ref-audio",p["audio"],"--ref-text",ref,"--gen-text",text,"--speed",str(self.speed.get()),"--force-nfe",str(self.steps.get()),"--ort-provider",provider,"--output-path",str(out)]
        self.go.configure(state="disabled",text="กำลังสร้าง..."); self.started=time.time(); self.status.configure(text="กำลังประมวลผล...",text_color="#fbbf24")
        threading.Thread(target=self.worker,args=(args,out),daemon=True).start()
    def worker(self,args,out):
        try:
            proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            for line in proc.stdout:self.after(0,lambda x=line.strip():self.status.configure(text=x[-55:]))
            if proc.wait():raise RuntimeError("Engine exited with code "+str(proc.returncode))
            self.current=out; self.after(0,self.done)
        except Exception as e:self.after(0,lambda:self.fail(str(e)))
    def done(self):
        self.go.configure(state="normal",text="✨ สร้างเสียง"); self.status.configure(text=f"สำเร็จใน {time.time()-self.started:.1f} วินาที",text_color="#35deb4"); self.outlabel.configure(text=self.current.name)
    def fail(self,e):
        self.go.configure(state="normal",text="✨ สร้างเสียง"); self.status.configure(text="สร้างไม่สำเร็จ",text_color="#fb7185"); messagebox.showerror("Engine error",e)
    def play(self):
        if not self.current:return
        import winsound; winsound.PlaySound(str(self.current),winsound.SND_FILENAME|winsound.SND_ASYNC)
    def export(self):
        if not self.current:return
        p=filedialog.asksaveasfilename(defaultextension=".wav",initialfile=self.current.name,filetypes=[("WAV","*.wav")])
        if p:shutil.copy2(self.current,p)

if __name__=="__main__":
    if not engine_mode():App().mainloop()
