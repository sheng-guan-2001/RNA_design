#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--model-path',required=True)
p.add_argument('--tokenizer-path',required=True)
p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--config',type=Path,default=root/'configs/train_m4.json')
a=p.parse_args(); cfg=json.loads(a.config.read_text())
if a.output.exists() and any(a.output.iterdir()):sys.exit('Choose an empty output directory')
if not cfg:sys.exit('Missing original training hyperparameters')
if not a.model_path or not a.tokenizer_path:sys.exit('Set BASE_MODEL and TOKENIZER in local.env')
for f in (Path(a.model_path),Path(a.tokenizer_path),a.prepared/'09_four_objective_conditional_train.jsonl',a.prepared/'10_four_objective_conditional_val.jsonl',a.prepared/'06_target_scaler_train_only.json'):
 if not f.exists():sys.exit(f'Missing training input: {f}')
cmd=[sys.executable,'-m','conditional_model.train_m4','--model-path',str(Path(a.model_path).resolve()),'--tokenizer-path',str(Path(a.tokenizer_path).resolve()),'--train',str((a.prepared/'09_four_objective_conditional_train.jsonl').resolve()),'--val',str((a.prepared/'10_four_objective_conditional_val.jsonl').resolve()),'--scaler',str((a.prepared/'06_target_scaler_train_only.json').resolve()),'--output',str(a.output.resolve())]
for k,v in cfg.items():
 flag='--'+k.replace('_','-')
 if isinstance(v,bool):
  if v:cmd.append(flag)
 elif v is not None:cmd.extend([flag,str(v)])
print('Training argv:',cmd,flush=True)
subprocess.run(cmd,cwd=root/'CodonTransformer-main',check=True)
