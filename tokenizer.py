import argparse,json
from pathlib import Path
from tokenizers import Tokenizer,models,pre_tokenizers,trainers

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="tokenizer"); p.add_argument("--vocab-size",type=int,default=32000); a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
tok=Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.train([a.input],trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,special_tokens=["<unk>","<bos>","<eos>"]))
tok.save(str(out/"tokenizer.json"))
(out/"metadata.json").write_text(json.dumps({"vocab_size":tok.get_vocab_size()},indent=2),encoding="utf-8")
