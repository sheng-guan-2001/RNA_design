"""M4 fixed-prefix BigBird training. Run: python -m conditional_model.train_m4 --help"""
import argparse,csv,json,math,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .config import DEFAULT_MODEL,DEFAULT_TRAIN,DEFAULT_VAL,DEFAULT_SCALER
from .dataset import ConditionalCDSDataset,load_vocab,load_scaler
from .collator_m4 import FixedPrefixCollator
from .model_m4 import FixedPrefixGlobalCodonTransformer

def parse():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
 p.add_argument('--tokenizer-path',type=Path,default=None)
 p.add_argument('--train',type=Path,default=DEFAULT_TRAIN)
 p.add_argument('--val',type=Path,default=DEFAULT_VAL)
 p.add_argument('--scaler',type=Path,default=DEFAULT_SCALER)
 p.add_argument('--output',type=Path,default=Path('/mnt/sdc/guansheng/protein_RNA/clear_understand/model_training/conditional_m4_fixed_prefix'))
 p.add_argument('--prefix-codons',type=int,default=50)
 p.add_argument('--suffix-mask-rate',type=float,default=1.0)
 p.add_argument('--cai-scale',type=float,default=1.0)
 p.add_argument('--global-mfe-scale',type=float,default=2.0)
 p.add_argument('--epochs',type=int,default=10)
 p.add_argument('--patience',type=int,default=10)
 p.add_argument('--batch-size',type=int,default=2)
 p.add_argument('--accumulation',type=int,default=8)
 p.add_argument('--backbone-lr',type=float,default=1e-5)
 p.add_argument('--condition-lr',type=float,default=1e-4)
 p.add_argument('--weight-decay',type=float,default=.01)
 p.add_argument('--warmup-ratio',type=float,default=.05)
 p.add_argument('--condition-hidden',type=int,default=128)
 p.add_argument('--seed',type=int,default=2026)
 p.add_argument('--organism-id',type=int,default=51)
 p.add_argument('--num-workers',type=int,default=0)
 p.add_argument('--precision',choices=['auto','bf16','fp16','fp32'],default='auto')
 p.add_argument('--attention-type',choices=['original_full','block_sparse'],default='block_sparse')
 p.add_argument('--save-every-epoch',action='store_true')
 p.add_argument('--freeze-backbone',action='store_true')
 p.add_argument('--limit-train',type=int,default=0)
 p.add_argument('--limit-val',type=int,default=0)
 p.add_argument('--allow-cpu',action='store_true')
 p.add_argument('--resume',type=Path,default=None)
 p.add_argument('--max-grad-norm',type=float,default=1.)
 return p.parse_args()

def main():
 a=parse()
 if not (0<a.suffix_mask_rate<=1):raise ValueError('Invalid suffix mask rate')
 if not torch.cuda.is_available() and not a.allow_cpu:raise RuntimeError('No GPU, --allow-cpu for tiny smoke only')
 if a.resume and not a.resume.exists():raise FileNotFoundError(a.resume)
 from transformers import BigBirdForMaskedLM
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
 a.output.mkdir(parents=True,exist_ok=True)
 tok=a.tokenizer_path or a.model_path
 vocab=load_vocab(tok)
 scaler=load_scaler(a.scaler)
 tr=ConditionalCDSDataset(a.train,scaler,vocab,organism_id=a.organism_id,limit=a.limit_train)
 va=ConditionalCDSDataset(a.val,scaler,vocab,organism_id=a.organism_id,limit=a.limit_val)
 # No train/test leakage. Proteins shorter than the frozen prefix have no trainable suffix.
 def only_suff(ds):
  eligible=[i for i,item in enumerate(ds.items) if len(item['input_ids'])-3>a.prefix_codons]
  print(f'[DATA] prefix={a.prefix_codons} codons, kept {len(eligible)}/{len(ds)}; skipped={len(ds)-len(eligible)}',flush=True)
  if not eligible:raise RuntimeError('No trainable sequences after length filter')
  return Subset(ds,eligible)
 tr=only_suff(tr);va=only_suff(va)
 tc=FixedPrefixCollator(vocab,a.prefix_codons,a.suffix_mask_rate)
 vc=FixedPrefixCollator(vocab,a.prefix_codons,a.suffix_mask_rate,seed=271828)
 tr_dl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,collate_fn=tc,num_workers=a.num_workers,pin_memory=torch.cuda.is_available())
 va_dl=DataLoader(va,batch_size=a.batch_size,shuffle=False,collate_fn=vc,num_workers=0,pin_memory=torch.cuda.is_available())
 print(f'[LOAD] {a.model_path}',flush=True)
 base=BigBirdForMaskedLM.from_pretrained(str(a.model_path),local_files_only=True)
 if base.config.vocab_size!=len(vocab):raise ValueError('Tokenizer/checkpoint mismatch')
 if a.organism_id>=base.config.type_vocab_size:raise ValueError('Organism out of range')
 base.bert.set_attention_type(a.attention_type)
 model=FixedPrefixGlobalCodonTransformer(base,a.condition_hidden,a.cai_scale,a.global_mfe_scale)
 if a.freeze_backbone:
  for p in model.backbone.parameters():p.requires_grad=False
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(dev)
 groups=[]
 bg=[p for p in model.backbone.parameters() if p.requires_grad]
 if bg:groups.append({'params':bg,'lr':a.backbone_lr})
 groups.append({'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad],
                'lr':a.condition_lr})
 opt=torch.optim.AdamW(groups,weight_decay=a.weight_decay)
 step_epoch=math.ceil(len(tr_dl)/a.accumulation)
 total_steps=step_epoch*a.epochs
 warmup=max(1,int(total_steps*a.warmup_ratio))
 from torch.optim.lr_scheduler import LambdaLR
 def lr_schedule(s):
  if s<warmup:return max(.001,s/max(1,warmup))
  t=min(1,(s-warmup)/max(1,total_steps-warmup))
  return max(.05,.5*(1+math.cos(math.pi*t)))
 sched=LambdaLR(opt,lr_schedule)
 precision=a.precision
 if precision=='auto':precision='bf16' if dev.type=='cuda' and torch.cuda.is_bf16_supported() else ('fp16' if dev.type=='cuda' else 'fp32')
 if dev.type=='cpu':precision='fp32'
 dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[precision]
 scaler_amp=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda' and precision=='fp16'))
 def autocast():return torch.autocast(device_type=dev.type,dtype=dtype,enabled=precision!='fp32')
 start=0;best=float('inf');bad=0;global_step=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False)
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
  if ck.get('gradscaler'):scaler_amp.load_state_dict(ck['gradscaler'])
  start=ck['epoch']+1;best=ck['best'];bad=ck['bad'];global_step=ck['step']
  print(f'[RESUME] start epoch {start+1}',flush=True)
 cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
 (a.output/'training_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
 metrics=a.output/'metrics.csv'
 if not a.resume or not metrics.exists():
  with metrics.open('w',newline='') as f:csv.writer(f).writerow(['epoch','train_ce','val_ce','global_step','cai_gate','mfe_gate','cai_scale','global_mfe_scale'])
 for epoch in range(start,a.epochs):
  model.train();opt.zero_grad(set_to_none=True)
  tot=0.;count=0
  for b,batch in enumerate(tr_dl):
   batch={k:v.to(dev,non_blocking=True) for k,v in batch.items()}
   rem=len(tr_dl)-(b//a.accumulation)*a.accumulation
   denom=min(rem,a.accumulation)
   with autocast():loss=model(**batch).loss
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   bs=batch['targets'].shape[0];tot+=loss.item()*bs;count+=bs
   scaler_amp.scale(loss/denom).backward()
   if (b+1)%a.accumulation==0 or b+1==len(tr_dl):
    scaler_amp.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),a.max_grad_norm)
    scaler_amp.step(opt);scaler_amp.update();sched.step();opt.zero_grad(set_to_none=True);global_step+=1
   if (b+1)%100==0:print(f'[EPOCH {epoch+1}] {b+1}/{len(tr_dl)} loss={loss.item():.4f}',flush=True)
  model.eval();vsum=0.;vn=0
  vc.generator.manual_seed(271828)
  with torch.inference_mode():
   for batch in va_dl:
    batch={k:v.to(dev,non_blocking=True) for k,v in batch.items()}
    with autocast():l=model(**batch).loss
    n=batch['targets'].shape[0];vsum+=l.item()*n;vn+=n
  tr_loss=tot/count;va_loss=vsum/vn
  print(f'[EPOCH {epoch+1}/{a.epochs}] train={tr_loss:.5f} val={va_loss:.5f} CAI_gate={model.cai_gate.item():.5f} MFE_gate={model.mfe_gate.item():.5f}',flush=True)
  with metrics.open('a',newline='') as f:csv.writer(f).writerow([epoch+1,tr_loss,va_loss,global_step,model.cai_gate.item(),model.mfe_gate.item(),a.cai_scale,a.global_mfe_scale])
  if a.save_every_epoch:
   model.save_bundle(a.output/'epochs'/f'epoch_{epoch+1:02d}',scaler,tok,cfg)
  if va_loss<best-1e-5:
   best=va_loss;bad=0;model.save_bundle(a.output/'best',scaler,tok,cfg)
  else:bad+=1
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
              'scheduler':sched.state_dict(),'gradscaler':scaler_amp.state_dict(),
              'epoch':epoch,'best':best,'bad':bad,'step':global_step},a.output/'last_checkpoint.pt')
  if bad>=a.patience:print('[EARLY STOP]',flush=True);break
 print(f'[DONE] best val CE={best:.5f}',flush=True)
if __name__=='__main__':main()
