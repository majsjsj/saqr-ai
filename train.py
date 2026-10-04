import argparse,json,math,random
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset,DataLoader
from model import SAQR,SAQRConfig

class MemmapDataset(Dataset):
    def __init__(self,path,block,start,end):
        self.path=path; self.block=block; self.start=int(start); self.end=int(end); self.data=None
    def _open(self):
        if self.data is None: self.data=np.memmap(self.path,dtype=np.uint32,mode="r")
    def __len__(self): return max(0,self.end-self.start-self.block-1)
    def __getitem__(self,i):
        self._open(); j=self.start+i
        x=torch.from_numpy(np.asarray(self.data[j:j+self.block],dtype=np.int64))
        y=torch.from_numpy(np.asarray(self.data[j+1:j+self.block+1],dtype=np.int64))
        return x,y

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def lr(step,c):
    if step<c["warmup_steps"]: return c["learning_rate"]*(step+1)/c["warmup_steps"]
    r=(step-c["warmup_steps"])/max(1,c["max_steps"]-c["warmup_steps"]); r=min(1,max(0,r))
    return c["min_lr"]+0.5*(1+math.cos(math.pi*r))*(c["learning_rate"]-c["min_lr"])

@torch.no_grad()
def evaluate(model,loader,device,n=20):
    model.eval(); vals=[]
    for i,(x,y) in enumerate(loader):
        if i>=n: break
        _,loss=model(x.to(device),y.to(device)); vals.append(loss.item())
    model.train(); return sum(vals)/max(1,len(vals))

p=argparse.ArgumentParser()
p.add_argument("--config",required=True); p.add_argument("--data",required=True)
p.add_argument("--tokenizer",default="tokenizer/tokenizer.json"); p.add_argument("--out",default="checkpoints")
a=p.parse_args(); c=json.loads(Path(a.config).read_text()); seed(c["seed"])
device="cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda": torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
tok=Tokenizer.from_file(a.tokenizer); c["vocab_size"]=tok.get_vocab_size()
mc=SAQRConfig(vocab_size=c["vocab_size"],block_size=c["block_size"],n_layer=c["n_layer"],n_head=c["n_head"],n_embd=c["n_embd"],dropout=c["dropout"],bias=c["bias"])
model=SAQR(mc).to(device); print("device:",device); print("parameters:",sum(p.numel() for p in model.parameters()))
token_count=Path(a.data).stat().st_size//4
if token_count<c["block_size"]*100: raise RuntimeError("Token dataset is too small.")
val_tokens=max(c["block_size"]*20,int(token_count*.01)); split=token_count-val_tokens
tr=MemmapDataset(a.data,c["block_size"],0,split); va=MemmapDataset(a.data,c["block_size"],split,token_count)
if len(tr)<100 or len(va)<20: raise RuntimeError("Dataset split is too small.")
tl=DataLoader(tr,batch_size=c["batch_size"],shuffle=True,drop_last=True,pin_memory=device=="cuda",num_workers=0)
vl=DataLoader(va,batch_size=c["batch_size"],shuffle=False,pin_memory=device=="cuda",num_workers=0)
opt=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"],betas=(.9,.95))
scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda"); Path(a.out).mkdir(parents=True,exist_ok=True); it=iter(tl)
for step in range(c["max_steps"]):
    opt.zero_grad(set_to_none=True); total=0.0
    for _ in range(c["grad_accum_steps"]):
        try: x,y=next(it)
        except StopIteration: it=iter(tl); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
            _,loss=model(x,y); loss=loss/c["grad_accum_steps"]
        scaler.scale(loss).backward(); total+=loss.item()
    rate=lr(step,c)
    for g in opt.param_groups: g["lr"]=rate
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
    if step%50==0: print(f"step={step} loss={total:.4f} lr={rate:.3e}",flush=True)
    if step%c["eval_interval"]==0 and step>0: print(f"validation_loss={evaluate(model,vl,device):.4f}",flush=True)
    if step%c["save_interval"]==0:
        ck={"model":model.state_dict(),"config":mc.__dict__,"step":step}; torch.save(ck,Path(a.out)/f"step_{step}.pt"); torch.save(ck,Path(a.out)/"last.pt")
print("training_complete")
