import argparse
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

p=argparse.ArgumentParser()
p.add_argument("--input",required=True); p.add_argument("--tokenizer",required=True)
p.add_argument("--output",default="data/tokens.bin"); p.add_argument("--chunk-chars",type=int,default=1000000)
a=p.parse_args()
tok=Tokenizer.from_file(a.tokenizer); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
total=0; buffer=[]; chars=0

def flush():
    global buffer,total,chars
    if not buffer: return
    ids=[]
    for text in buffer: ids.extend(tok.encode(text).ids)
    if ids:
        np.asarray(ids,dtype=np.uint32).tofile(out.open("ab")); total+=len(ids)
    buffer=[]; chars=0; print(f"tokens_written={total}",flush=True)

with open(a.input,"r",encoding="utf-8",errors="ignore") as f:
    for line in f:
        buffer.append(line); chars+=len(line)
        if chars>=a.chunk_chars: flush()
    flush()
print(f"done: {total} tokens -> {out}")
