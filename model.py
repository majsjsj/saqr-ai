import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SAQRConfig:
    vocab_size:int=32000; block_size:int=1024; n_layer:int=12
    n_head:int=12; n_embd:int=768; dropout:float=0.0; bias:bool=False

class CausalSelfAttention(nn.Module):
    def __init__(self,c):
        super().__init__(); assert c.n_embd%c.n_head==0
        self.n_head=c.n_head; self.head_dim=c.n_embd//c.n_head
        self.qkv=nn.Linear(c.n_embd,3*c.n_embd,bias=c.bias)
        self.proj=nn.Linear(c.n_embd,c.n_embd,bias=c.bias)
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x):
        b,t,ch=x.shape; q,k,v=self.qkv(x).split(ch,2)
        q=q.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop.p if self.training else 0.0,is_causal=True)
        return self.drop(self.proj(y.transpose(1,2).contiguous().view(b,t,ch)))

class MLP(nn.Module):
    def __init__(self,c):
        super().__init__(); h=4*c.n_embd
        self.fc=nn.Linear(c.n_embd,h,bias=c.bias); self.proj=nn.Linear(h,c.n_embd,bias=c.bias); self.drop=nn.Dropout(c.dropout)
    def forward(self,x): return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.ln1=nn.LayerNorm(c.n_embd); self.attn=CausalSelfAttention(c); self.ln2=nn.LayerNorm(c.n_embd); self.mlp=MLP(c)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))

class SAQR(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.token_embedding=nn.Embedding(c.vocab_size,c.n_embd); self.position_embedding=nn.Embedding(c.block_size,c.n_embd)
        self.drop=nn.Dropout(c.dropout); self.blocks=nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f=nn.LayerNorm(c.n_embd); self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=False)
        self.lm_head.weight=self.token_embedding.weight; self.apply(self._init)
        for n,p in self.named_parameters():
            if n.endswith("proj.weight"): nn.init.normal_(p,0.0,0.02/math.sqrt(2*c.n_layer))
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0.0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.config.block_size: raise ValueError("Sequence exceeds block_size")
        pos=torch.arange(t,device=idx.device)
        x=self.drop(self.token_embedding(idx)+self.position_embedding(pos))
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.ln_f(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=0.8,top_k=50):
        for _ in range(max_new_tokens):
            logits,_=self(idx[:,-self.config.block_size:]); logits=logits[:,-1]/max(temperature,1e-5)
            if top_k:
                v,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<v[:,-1:]]=float("-inf")
            idx=torch.cat((idx,torch.multinomial(F.softmax(logits,dim=-1),1)),1)
        return idx
