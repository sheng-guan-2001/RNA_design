#!/usr/bin/env python3
"""Build a reviewable V3 CLI command from the actual retained generator --help."""
import argparse,json,re,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--module',choices=['m4_q95_benchmark.generate_q95_m4','m4_extreme_cai08.generate_q95_m4'],required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--prefixes',type=Path,required=True)
p.add_argument('--targets',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--manifest',type=Path,help='Original generation manifest for recorded hyperparameters')
p.add_argument('--execute',action='store_true',help='Execute after checking the printed command')
p.add_argument('extra',nargs=argparse.REMAINDER,help='Extra original options after --')
a=p.parse_args()
cwd=root/'CodonTransformer-main'
for f in (a.checkpoint,a.prefixes,a.targets,a.reference):
 if not f.exists():sys.exit(f'Missing input: {f}')
result=subprocess.run([sys.executable,'-m',a.module,'--help'],cwd=cwd,text=True,capture_output=True)
if result.returncode:sys.exit(result.stdout+result.stderr)
flags=set(re.findall(r'--[a-zA-Z][a-zA-Z0-9_-]*',result.stdout))
cmd=[sys.executable,'-m',a.module]; used=set()
def option(names,value,required=True):
 for name in names:
  if name in flags:
   cmd.extend([name,str(value)]);used.add(name);return True
 if required:sys.exit('Unrecognized generator CLI option: '+str(names)+'\n'+result.stdout)
 return False
option(['--checkpoint','--checkpoint-dir','--model-path'],a.checkpoint.resolve())
option(['--local-csv','--prefix-csv','--prefixes-csv','--local-prefix-csv','--local-prefixes-csv','--optimized-csv','--local-prefixes','--prefixes','--local'],a.prefixes.resolve())
option(['--targets-json','--target-json','--quantile-targets-json','--quantile-json','--targets','--quantiles-json'],a.targets.resolve())
if not option(['--reference-csv','--test-csv','--reference','--test'],a.reference.resolve(),False):
 print('WARNING: no recognized reference flag; inspect generator defaults.',flush=True)
if not option(['--output','--output-csv'],a.output.resolve(),False):
 option(['--outdir','--output-dir'],a.output.resolve().parent)
 print('WARNING: generator uses an output directory; check generated filename.',flush=True)
# Only reuse recognized scalar sampling knobs; do not guess missing values.
allowed=['seed','temperature','replicates','n_per_protein','n_candidates','samples_per_prefix','replicates_per_prefix','gpu_batch_size','batch_size','max_abs_z','prefix_codons','organism_id','n_proteins','max_proteins','device','top_k','top_p']
settings={}
if a.manifest:
 obj=json.loads(a.manifest.read_text())
 for d in [obj]+[obj[k] for k in ['args','config','settings','generation_settings'] if isinstance(obj.get(k),dict)]:
  for k in allowed:
   if k in d and isinstance(d[k],(str,int,float)) and not isinstance(d[k],bool):settings[k]=d[k]
for k,v in settings.items():
 flag='--'+k.replace('_','-')
 if flag in flags and flag not in used:cmd.extend([flag,str(v)])
extra=a.extra[1:] if a.extra and a.extra[0]=='--' else a.extra
cmd.extend(extra)
print('Review this command (unrecorded options use the generator defaults):',flush=True)
print(shlex.join(cmd),flush=True)
if not a.execute:sys.exit(0)
if a.output.exists():sys.exit('Output already exists; choose a new path')
a.output.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(cmd,cwd=cwd,check=True)
