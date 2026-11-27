"""Score generated M4 CDS with exact same reference CAI and ViennaRNA definitions.
Run: python -m conditional_model.score_m4 --generated ... --output ... --workers 12
"""
import argparse,concurrent.futures,math,os
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from .evaluate import metrics,cai_model,DEFAULT_FASTA

_MODEL=None

def init(fasta):
 global _MODEL
 for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[var]='1'
 _MODEL=cai_model(fasta)

def score(x):
 i,dna=x
 try:
  f=metrics(dna,_MODEL,37.,150,100)
  return i,f,None
 except Exception as e:return i,None,repr(e)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--generated',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--heg-fasta',type=Path,default=Path(DEFAULT_FASTA))
 p.add_argument('--workers',type=int,default=12)
 a=p.parse_args()
 df=pd.read_csv(a.generated,dtype={'test_id':str,'generated_dna':str,'fixed_prefix_150nt':str})
 if df.empty:raise RuntimeError('No generated sequences')
 for i,r in df.iterrows():
  dna=str(r['generated_dna']);protein=str(r['protein_sequence'])
  if not dna.startswith(str(r['fixed_prefix_150nt'])):raise RuntimeError(f'Prefix changed at row {i}')
  if str(Seq(dna).translate(table=11,cds=True))!=protein:raise RuntimeError(f'Protein changed at row {i}')
 df['actual_CAI']=float('nan');df['actual_Global_MFE']=float('nan')
 df['actual_MFE30']=float('nan');df['actual_OE20']=float('nan')
 df['score_error']=''
 uniq={dna:i for i,dna in enumerate(df['generated_dna'].unique())}
 seqs=[(i,dna) for dna,i in uniq.items()]
 vals={}
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.heg_fasta),)) as pool:
  for n,(i,f,err) in enumerate(pool.map(score,seqs,chunksize=4),1):
   vals[i]=(f,err)
   if n%50==0 or n==len(seqs):print(f'[SCORE] {n}/{len(seqs)} unique CDS',flush=True)
 for ix,dna in enumerate(df['generated_dna']):
  f,err=vals[uniq[dna]]
  if err:df.at[ix,'score_error']=err;continue
  for k in ['CAI','Global_MFE','MFE30','OE20']:df.at[ix,'actual_'+k]=float(f[k])
 for k in ['CAI','Global_MFE']:
  df[f'abs_error_{k}']=(df[f'actual_{k}']-df[f'target_{k}']).abs()
 df['CAI_delta_from_matched']=float('nan');df['MFE_delta_from_matched']=float('nan')
 # mean across Matched replicates for each fixed prefix / local condition
 group_cols=['test_id','local_scenario']
 for _,g in df.groupby(group_cols):
  matched=g[g['global_scenario']=='matched']
  if matched.empty:continue
  avg_c=matched['actual_CAI'].mean();avg_m=matched['actual_Global_MFE'].mean()
  df.loc[g.index,'CAI_delta_from_matched']=g['actual_CAI']-avg_c
  df.loc[g.index,'MFE_delta_from_matched']=g['actual_Global_MFE']-avg_m
 df['local_MFE30_spread']=df.groupby(group_cols)['actual_MFE30'].transform(lambda s:s.max()-s.min())
 df['local_OE20_spread']=df.groupby(group_cols)['actual_OE20'].transform(lambda s:s.max()-s.min())
 a.output.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(a.output,index=False)
 summary=df.groupby(['local_scenario','global_scenario'],as_index=False).agg(
  n=('actual_CAI','count'),CAI_MAE=('abs_error_CAI','mean'),Global_MFE_MAE=('abs_error_Global_MFE','mean'),
  CAI_delta=('CAI_delta_from_matched','mean'),Global_MFE_delta=('MFE_delta_from_matched','mean'),
  MFE30_spread=('local_MFE30_spread','mean'),OE20_spread=('local_OE20_spread','mean'))
 summ=a.output.with_name(a.output.stem+'_summary.csv')
 summary.to_csv(summ,index=False)
 failed_count = int(df['score_error'].ne('').sum())
 print(f'[DONE] scored={a.output}; summary={summ}; failed={failed_count}', flush=True)
if __name__=='__main__':main()
