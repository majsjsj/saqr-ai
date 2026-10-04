import argparse,torch
from tokenizers import Tokenizer
from model import SAQR,SAQRConfig

p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",required=True); p.add_argument("--tokenizer",default="tokenizer/tokenizer.json"); p.add_argument("--tokens",type=int,default=100); p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=50); a=p.parse_args()
device="cuda" if torch.cuda.is_available() else "cpu"; ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
model=SAQR(SAQRConfig(**ck["config"])).to(device); model.load_state_dict(ck["model"]); model.eval()
tok=Tokenizer.from_file(a.tokenizer); x=torch.tensor([tok.encode(a.prompt).ids],device=device)
print(tok.decode(model.generate(x,a.tokens,a.temperature,a.top_k)[0].tolist()))
