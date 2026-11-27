#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Codon-constrained MFE30 prefix enumeration + OE20 simulated annealing.

No neural training. For each target protein and MFE30/OE20 High/Low pair:
  1) enumerate synonyms for codons 2..10 (start is unchanged), score RNAfold first 30 nt;
  2) retain diverse prefixes nearest target MFE30;
  3) lock codons 1..10, simulated anneal synonymous codons 11..50 on OE20
     computed exactly with RNAplfold -W 150 -L 100 -u 30 -T 37.
Intermediate files per protein/scenario are atomic and resumable.
"""
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, itertools, json, math, os, random, shutil, subprocess, tempfile, time, traceback
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_DEFAULT='/mnt/sdc/guansheng/protein_RNA/clear_understand/data_preparation/ecoli_codontransformer/14_reserved_test_four_objectives.csv'
OUT_DEFAULT='/mnt/sdc/guansheng/protein_RNA/clear_understand/model_training/OE20_range/two_stage_search'
R=0.00198720425864083
SYN={
 'A':'GCT GCC GCA GCG','R':'CGT CGC CGA CGG AGA AGG','N':'AAT AAC',
 'D':'GAT GAC','C':'TGT TGC','Q':'CAA CAG','E':'GAA GAG',
 'G':'GGT GGC GGA GGG','H':'CAT CAC','I':'ATT ATC ATA',
 'L':'TTA TTG CTT CTC CTA CTG','K':'AAA AAG','M':'ATG',
 'F':'TTT TTC','P':'CCT CCC CCA CCG','S':'TCT TCC TCA TCG AGT AGC',
 'T':'ACT ACC ACA ACG','W':'TGG','Y':'TAT TAC','V':'GTT GTC GTA GTG'}
SYN={x:tuple(v.split()) for x,v in SYN.items()}
SCENARIOS=('mfe_high__oe_high','mfe_high__oe_low','mfe_low__oe_high','mfe_low__oe_low')


def cli():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--input',default=SOURCE_DEFAULT)
 p.add_argument('--output',default=OUT_DEFAULT)
 p.add_argument('--max-proteins',type=int,default=0,help='0 means all eligible proteins')
 p.add_argument('--workers',type=int,default=8,help='processes; each runs RNAplfold independently')
 p.add_argument('--prefix-topk',type=int,default=5,help='number of distinct prefix solutions kept per MFE target')
 p.add_argument('--prefix-pool',type=int,default=300,help='candidate pool before diversity-aware selection')
 p.add_argument('--enumeration-cap',type=int,default=20000,help='hard cap on 30-nt candidates per protein; if exceeded use sampled search')
 p.add_argument('--restarts',type=int,default=2,help='SA repeats per prefix')
 p.add_argument('--steps',type=int,default=60,help='RNAplfold evaluations per SA restart, approximately')
 p.add_argument('--random-baseline',type=int,default=60,help='random candidates per prefix for fair budget comparison; 0 disables')
 p.add_argument('--scenarios',default='mfe_high__oe_low,mfe_low__oe_high',help='comma-delimited scenario names')
 p.add_argument('--delta-z',type=float,default=0.75)
 p.add_argument('--mfe-std',type=float,default=2.10996)
 p.add_argument('--oe-std',type=float,default=2.34794)
 p.add_argument('--seed',type=int,default=2026)
 p.add_argument('--temp-start',type=float,default=0.50,help='initial annealing temperature in units of standardized target error')
 p.add_argument('--temp-end',type=float,default=0.015)
 p.add_argument('--edits-max',type=int,default=3,help='max number of codon replacements per proposal')
 p.add_argument('--rna-timeout',type=float,default=45)
 p.add_argument('--force',action='store_true')
 p.add_argument('--summary-only',action='store_true')
 return p.parse_args()


def load_records(path,n):
 from Bio.Seq import Seq
 df=pd.read_csv(path,dtype={'dna_sequence':str,'protein_sequence':str},low_memory=False)
 records=[];seen=set()
 for _,r in df.iterrows():
  dna=str(r['dna_sequence']).strip().upper().replace('U','T')
  prot=str(r['protein_sequence']).strip().upper().rstrip('*_')
  if len(prot)<10 or prot in seen or len(dna)!=3*(len(prot)+1) or dna[:3] not in ('ATG','GTG','TTG') or dna[-3:] not in ('TAA','TAG','TGA'):continue
  if any(a not in SYN for a in prot):continue
  if str(Seq(dna).translate(table=11,cds=True))!=prot:continue
  seen.add(prot)
  records.append(dict(id=str(r['record_id']),dna=dna,prot=prot,oe=float(r['OE20']),mfe=float(r['MFE30'])))
 if n>0:records=records[:n]
 return records


def actual_mfe30(dna):
 import RNA
 return float(RNA.fold(dna[:30].replace('T','U'))[1])


def score_oe20(dna,timeout=45):
 # Full CDS is used, preserving the exact benchmark semantics even for short CDS.
 with tempfile.TemporaryDirectory(prefix='two_stage_oe_') as td:
  command=['RNAplfold','-W','150','-L','100','-u','30','-T','37']
  r=subprocess.run(command,input='>candidate\n'+dna.replace('T','U')+'\n',cwd=td,capture_output=True,text=True,timeout=timeout)
  if r.returncode:raise RuntimeError('RNAplfold: '+r.stderr[-400:])
  files=list(Path(td).glob('*_lunp'))
  if len(files)!=1:raise RuntimeError(f'Expected one _lunp, found {len(files)}')
  probability=None
  for line in files[0].read_text().splitlines():
   cells=line.split()
   if cells and cells[0]=='20':
    if len(cells)<=20:raise RuntimeError('No joint-unpaired probability for 20nt')
    probability=float(cells[20]);break
  if probability is None or not (0<probability<=1):raise RuntimeError(f'Invalid OE20 probability: {probability}')
  return -R*(273.15+37)*math.log(probability)


def prefix_candidates(record,cap,seed,pool_count):
 """Exact enumeration when feasible; otherwise reproducible uniform sampling.
 Return a scored pool sorted by sequence, count and whether exhaustive.
 """
 dna,prot=record['dna'],record['prot']
 opts=[[dna[:3]]] + [list(SYN[a]) for a in prot[1:10]]
 total=math.prod(len(v) for v in opts)
 rng=random.Random(seed)
 if total<=cap:
  iterator=(''.join(t) for t in itertools.product(*opts))
  exhaustive=True
 else:
  # Include the original prefix; use uniform random combinations, no false claim of enumeration.
  chosen={dna[:30]}
  attempts=0
  while len(chosen)<cap and attempts<max(cap*6,1000):
   attempts+=1
   chosen.add(''.join(rng.choice(v) for v in opts))
  iterator=iter(sorted(chosen))
  exhaustive=False
 # MFE30 folding is cheap, but avoid holding huge decoded sequences in RAM if cap is high.
 out=[]
 for p in iterator:
  out.append((p,actual_mfe30(p)))
 return out,total,exhaustive


def select_diverse(scored,target,k,pool):
 """Target-near candidate pool; select diversity by Hamming codon distance."""
 ordered=sorted(scored,key=lambda t:(abs(t[1]-target),t[0]))
 eligible=[x for x in ordered if abs(x[1]-target)<=0.5][:max(k,pool)]
 if len(eligible)<k:eligible=ordered[:max(k,pool)]
 if not eligible:return []
 selected=[eligible[0]];taken={eligible[0][0]}
 while len(selected)<k and len(taken)<len(eligible):
  # Minimum normalized codon distance, then absolute target error (bounded diversity priority)
  def rank(it):
   p,m=it
   mind=min(sum(p[3*j:3*j+3]!=sp[3*j:3*j+3] for j in range(10)) for sp,_ in selected)
   return (mind,-abs(m-target),p)
  rest=[it for it in eligible if it[0] not in taken]
  x=max(rest,key=rank)
  selected.append(x);taken.add(x[0])
 return selected


def eligible_suffix(dna,prot):
 result={}
 for i in range(10,min(50,len(prot))):
  old=dna[3*i:3*i+3]
  choices=tuple(x for x in SYN[prot[i]] if x!=old)
  if choices:result[i]=choices
 return result


def update_seq(current,protein,edit_positions,rng,max_edits):
 if not edit_positions:return current
 keys=list(edit_positions)
 indices=rng.sample(keys,rng.randint(1,min(max_edits,len(keys))))
 parts=list(current)
 for i in indices:
  old=''.join(parts[3*i:3*i+3])
  choices=[x for x in SYN[protein[i]] if x!=old]
  if choices:parts[3*i:3*i+3]=rng.choice(choices)
 return ''.join(parts)


def optimize_sa(start,protein,target,ref_oe,steps,rng,temperature_start,temperature_end,max_edits,timeout,cache):
 # Fitness is distance to a requested value, not unconstrained minimization.
 options=eligible_suffix(start,protein)
 def evaluate(seq):
  key=seq[:150] if len(seq)>=150 else seq
  if key not in cache:
   cache[key]=score_oe20(seq,timeout)
  return cache[key]
 current=start;cur_oe=evaluate(current);cur_loss=abs(cur_oe-target)/2.34794
 best=(cur_loss,cur_oe,current)
 for j in range(steps):
  proposal=update_seq(current,protein,options,rng,max_edits)
  if proposal==current:break
  oe=evaluate(proposal);loss=abs(oe-target)/2.34794
  fraction=j/max(steps-1,1)
  temp=temperature_start*(temperature_end/temperature_start)**fraction
  if loss<cur_loss or rng.random()<math.exp(min(0,(cur_loss-loss)/max(temp,1e-10))):
   current,cur_oe,cur_loss=proposal,oe,loss
  if loss<best[0]:best=(loss,oe,proposal)
 return best


def optimize_random(start,protein,target,n,rng,timeout,cache):
 options=eligible_suffix(start,protein)
 def evaluate(seq):
  key=seq[:150] if len(seq)>=150 else seq
  if key not in cache:cache[key]=score_oe20(seq,timeout)
  return cache[key]
 val=evaluate(start);best=(abs(val-target)/2.34794,val,start)
 for _ in range(n):
  # Random candidate draws 1..8 edits from original prefix for fair evaluation budget.
  seq=update_seq(start,protein,options,rng,8)
  if seq==start:break
  val=evaluate(seq);loss=abs(val-target)/2.34794
  if loss<best[0]:best=(loss,val,seq)
 return best


def run_one(job):
 r,a,scenario=job
 out=Path(a['output'])/'per_protein';out.mkdir(parents=True,exist_ok=True)
 token=scenario.replace('/','_');filename=out/f"{r['id']}_{token}.json"
 if filename.exists() and not a['force']:
  return json.loads(filename.read_text())
 seed=int(hashlib.sha256(f"{a['seed']}:{r['id']}:{scenario}".encode()).hexdigest()[:12],16)
 rng=random.Random(seed)
 m_direction=1 if 'mfe_high' in scenario else -1
 o_direction=1 if 'oe_high' in scenario else -1
 m_target=min(0.,r['mfe']+m_direction*a['delta_z']*a['mfe_std'])
 o_target=max(0.,r['oe']+o_direction*a['delta_z']*a['oe_std'])
 start_time=time.monotonic()
 scored,total,exhaustive=prefix_candidates(r,a['enumeration_cap'],seed,a['prefix_pool'])
 prefixes=select_diverse(scored,m_target,a['prefix_topk'],a['prefix_pool'])
 cache={}
 best_sa=None;best_rnd=None
 for pidx,(pref,mfe) in enumerate(prefixes):
  start=pref+r['dna'][30:]
  for rep in range(a['restarts']):
   result=optimize_sa(start,r['prot'],o_target,r['oe'],a['steps'],rng,a['temp_start'],a['temp_end'],a['edits_max'],a['rna_timeout'],cache)
   candidate={'method':'SA','prefix_index':pidx,'restart':rep,'dna':result[2], 'MFE30':mfe,'OE20':result[1]}
   candidate['mfe_error']=abs(mfe-m_target);candidate['oe_error']=abs(result[1]-o_target)
   candidate['combined_error']=candidate['mfe_error']/a['mfe_std']+candidate['oe_error']/a['oe_std']
   if best_sa is None or candidate['combined_error']<best_sa['combined_error']:best_sa=candidate
  if a['random_baseline']>0:
   # Per-prefix comparison; total random budget can differ from SA; recorded explicitly.
   result=optimize_random(start,r['prot'],o_target,a['random_baseline']*a['restarts'],rng,a['rna_timeout'],cache)
   candidate={'method':'Random','prefix_index':pidx,'restart':0,'dna':result[2], 'MFE30':mfe,'OE20':result[1]}
   candidate['mfe_error']=abs(mfe-m_target);candidate['oe_error']=abs(result[1]-o_target)
   candidate['combined_error']=candidate['mfe_error']/a['mfe_std']+candidate['oe_error']/a['oe_std']
   if best_rnd is None or candidate['combined_error']<best_rnd['combined_error']:best_rnd=candidate
 baseline_m=actual_mfe30(r['dna']);baseline_o=score_oe20(r['dna'],a['rna_timeout'])
 results={'test_id':r['id'],'scenario':scenario,'protein_length_aa':len(r['prot']),
  'prefix_search_space':total,'prefix_candidates_evaluated':len(scored),'prefix_exact_enumeration':exhaustive,
  'prefixes_retained':len(prefixes),'target_MFE30':m_target,'target_OE20':o_target,
  'reference_MFE30':baseline_m,'reference_OE20':baseline_o,
  'sa':best_sa,'random':best_rnd,'distinct_oe20_evaluations':len(cache)+1,
  'estimated_sa_proposals':len(prefixes)*a['restarts']*a['steps'],
  'estimated_random_proposals':len(prefixes)*a['random_baseline']*a['restarts'],
  'runtime_sec':round(time.monotonic()-start_time,3)}
 # Codon-consistency check for all outputs; preserve alternative bacterial initiators
 from Bio.Seq import Seq
 for selected in [best_sa,best_rnd]:
  if selected is None:continue
  seq=selected['dna']
  if str(Seq(seq).translate(table=11,cds=True))!=r['prot']:
   raise RuntimeError('Synonymous translation validation failed')
  if abs(actual_mfe30(seq)-selected['MFE30'])>1e-5:raise RuntimeError('MFE30 changed unexpectedly')
 temp=filename.with_suffix('.tmp');temp.write_text(json.dumps(results,ensure_ascii=False,indent=2));os.replace(temp,filename)
 return results


def summarize(output):
 out=Path(output);files=list((out/'per_protein').glob('*.json'))
 if not files:print('No results yet');return
 rows=[];sequences=[]
 for file in files:
  r=json.loads(file.read_text())
  for method,key in [('SA','sa'),('Random','random')]:
   item=r.get(key)
   if not item:continue
   row={'test_id':r['test_id'],'scenario':r['scenario'],'method':method,
        'prefix_space':r['prefix_search_space'],'prefix_candidates_evaluated':r['prefix_candidates_evaluated'],
        'prefix_exact':r['prefix_exact_enumeration'],'prefixes_retained':r['prefixes_retained'],
        'reference_MFE30':r['reference_MFE30'],'target_MFE30':r['target_MFE30'],
        'actual_MFE30':item['MFE30'],'MAE_MFE30':item['mfe_error'],
        'reference_OE20':r['reference_OE20'],'target_OE20':r['target_OE20'],
        'actual_OE20':item['OE20'],'MAE_OE20':item['oe_error'],
        'combined_error':item['combined_error'],'n_oe20_evaluations_total':r['distinct_oe20_evaluations'],
        'runtime_sec_total':r['runtime_sec'],
        'MFE30_hit':(item['MFE30']>r['reference_MFE30']+1e-5 if 'mfe_high' in r['scenario'] else item['MFE30']<r['reference_MFE30']-1e-5),
        'OE20_hit':(item['OE20']>r['reference_OE20']+1e-5 if 'oe_high' in r['scenario'] else item['OE20']<r['reference_OE20']-1e-5),
        'MFE30_within_0p5':item['mfe_error']<=0.5,'OE20_within_0p5':item['oe_error']<=0.5,
        'both_within_0p5':item['mfe_error']<=0.5 and item['oe_error']<=0.5}
   rows.append(row)
   sequences.append({'test_id':r['test_id'],'scenario':r['scenario'],'method':method,'dna_sequence':item['dna'],'MFE30':item['MFE30'],'OE20':item['OE20']})
 df=pd.DataFrame(rows);df.to_csv(out/'01_per_protein_results.csv',index=False)
 pd.DataFrame(sequences).to_csv(out/'02_best_sequences.csv',index=False)
 summary=df.groupby(['scenario','method'],as_index=False).agg(
  n_proteins=('test_id','nunique'),MFE30_MAE=('MAE_MFE30','mean'),OE20_MAE=('MAE_OE20','mean'),
  MFE30_direction_rate=('MFE30_hit','mean'),OE20_direction_rate=('OE20_hit','mean'),
  MFE30_success_0p5=('MFE30_within_0p5','mean'),OE20_success_0p5=('OE20_within_0p5','mean'),
  joint_success_0p5=('both_within_0p5','mean'),
  mean_prefix_candidates=('prefix_candidates_evaluated','mean'),mean_oe_evaluations=('n_oe20_evaluations_total','mean'),
  mean_runtime_sec=('runtime_sec_total','mean'))
 summary.to_csv(out/'03_summary.csv',index=False)
 print('\n[SUMMARY]\n'+summary.to_string(index=False))
 print('[FILES] '+str(out/'03_summary.csv'))
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(10,5));
  pv=summary.pivot(index='scenario',columns='method',values='joint_success_0p5').reindex(sorted(summary.scenario.unique()))
  pv.plot.bar(ax=ax);ax.set_ylim(0,1);ax.set_ylabel('Both targets within 0.5 kcal/mol');ax.set_title('Joint target attainment: SA vs random');fig.tight_layout();fig.savefig(out/'04_joint_success.png',dpi=180);plt.close(fig)
 except Exception as e:print('[PLOT WARNING]',e)


def main():
 a=cli();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.summary_only:summarize(a.output);return
 if not shutil.which('RNAplfold'):raise RuntimeError('RNAplfold not found. Activate protein_RNA environment.')
 try:import RNA
 except ImportError as e:raise RuntimeError('Python ViennaRNA package not found. pip/conda install ViennaRNA') from e
 scenarios=[x.strip() for x in a.scenarios.split(',') if x.strip()]
 if not scenarios or not set(scenarios).issubset(SCENARIOS):raise ValueError('Invalid scenarios '+str(scenarios))
 if a.prefix_topk<1 or a.enumeration_cap<1 or a.restarts<1 or a.steps<1 or a.workers<1:raise ValueError('Invalid positive argument')
 params={k:v for k,v in vars(a).items() if k not in ('force','summary_only')}
 conf=out/'00_config.json'
 if conf.exists() and not a.force:
  old=json.loads(conf.read_text())
  if old!=params:raise RuntimeError('Output directory contains a different run configuration; choose a new --output or --force')
 conf.write_text(json.dumps(params,indent=2))
 data=load_records(a.input,a.max_proteins)
 print(f'[DATA] {len(data)} proteins, scenarios={scenarios}, workers={a.workers}',flush=True)
 if not data:raise RuntimeError('No eligible protein')
 tasks=[(r,vars(a),s) for r in data for s in scenarios]
 errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(run_one,job):job for job in tasks}
  for i,fut in enumerate(concurrent.futures.as_completed(futures),1):
   job=futures[fut]
   try:
    result=fut.result()
    print(f"[{i}/{len(tasks)}] protein={result['test_id']} {result['scenario']} exact={result['prefix_exact_enumeration']} prefixes={result['prefix_candidates_evaluated']} OE calls={result['distinct_oe20_evaluations']} sec={result['runtime_sec']}",flush=True)
   except Exception as e:
    errors.append({'id':job[0]['id'],'scenario':job[2],'error':repr(e),'traceback':traceback.format_exc()})
    print(f'[ERROR] {job[0]["id"]} {job[2]}: {e}',flush=True)
 if errors:(out/'99_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 summarize(a.output)
 if errors:raise RuntimeError(f'{len(errors)} tasks failed, see 99_errors.json')

if __name__=='__main__':main()
