#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAI-aware two-stage prefix generation compatible with existing M4 generator.

Uses exactly the E. coli CAI reference fitted by conditional_model.evaluate.
First 30 nt: synonymous enumeration/sampling, MFE30 near target then CAI preference.
Codons 11..50: RNAplfold OE20 + CAI_150nt tradeoff by SA.

Outputs 02_q95_local_prefixes.csv and 00_train_quantile_targets.json,
compatible with m4_q95_benchmark.generate_q95_m4 and original score_m4.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from conditional_model.evaluate import cai_model, DEFAULT_FASTA
from OE20_range_experiment.two_stage_local_search import (
    SYN, prefix_candidates, score_oe20,
)

BASE = Path('/mnt/sdc/guansheng/protein_RNA/clear_understand')
TEST = BASE/'data_preparation/ecoli_codontransformer/14_reserved_test_four_objectives.csv'
OUT = BASE/'model_training/M4_CAIaware_CAI08_Q98'


def log_cai_for_codons(dna, index):
    """Weighted log CAI; full sequence must be a multiple of three, no stop."""
    n = len(dna)//3
    if not n or len(dna)%3: raise ValueError('Bad CAI segment length')
    return sum(math.log(max(float(index[dna[i:i+3]]), 1e-12))
               for i in range(0,len(dna),3))/n


def hamming_codons(a,b):
    return sum(a[i:i+3] != b[i:i+3] for i in range(0,min(len(a),len(b)),3))


def choose_prefixes(samples, target_mfe, index, cfg):
    """Keep local MFE30 feasible, prioritize CAI within a near-best MFE envelope.

    Uses adaptive tolerance relative to best achievable target error, not
    to an impossible absolute 0.00 target. Diversity required among 5 solutions.
    """
    vals=[]
    for seq, mfe in samples:
        e=abs(float(mfe)-target_mfe)
        c=math.exp(log_cai_for_codons(seq,index))
        vals.append((seq,float(mfe),e,c))
    if not vals: return []
    best_err=min(x[2] for x in vals)
    # Include near-optimal MFE30 candidates only. 0.5 kcal/mol beyond best.
    pool=[x for x in vals if x[2] <= best_err + cfg['mfe_slack']+1e-9]
    pool=sorted(pool,key=lambda x:(-x[3],x[2],x[0]))[:cfg['prefix_pool']]
    chosen=[]
    while len(chosen)<cfg['prefix_topk'] and pool:
        if not chosen:
            pick=pool.pop(0)
        else:
            # CAI is primary; modest bonus to codon diversity.
            pick=max(pool,key=lambda x:(x[3]+cfg['diversity_bonus']*min(hamming_codons(x[0],z[0]) for z in chosen)/10,
                                        -x[2],x[0]))
            pool.remove(pick)
        chosen.append(pick)
    return chosen


def fit_cai_index(fasta):
    model=cai_model(str(fasta))
    weights={str(k).upper().replace('U','T'):float(v) for k,v in model.items()}
    missing=[x for v in SYN.values() for x in v if x not in weights or weights[x]<=0]
    if missing:raise RuntimeError('Missing CAI codon weights: '+','.join(sorted(set(missing))))
    return weights


def mutate_suffix_30_to150(seq, protein, rng, max_edits):
    editable=[i for i in range(10,min(50,len(protein))) if len(SYN[protein[i]])>1]
    if not editable:return seq
    k=rng.randint(1,min(max_edits,len(editable)))
    idx=rng.sample(editable,k)
    a=list(seq)
    for i in idx:
        old=''.join(a[3*i:3*i+3])
        choices=[c for c in SYN[protein[i]] if c != old]
        if choices:a[3*i:3*i+3]=rng.choice(choices)
    return ''.join(a)


def optimize_cai_aware_sa(start,protein,target_oe,index,cfg,rng,cache):
    """Optimize 31-150 nt without changing 1-30 nt.

    OE20 loss uses a soft tolerance; once OE20 is sufficiently close,
    increasing CAI_150 becomes valuable. We keep a small OE penalty inside
    feasible region to avoid excessive drift.
    """
    def metrics(seq):
        key=seq[:150]
        if key not in cache:cache[key]=score_oe20(seq,timeout=cfg['rna_timeout'])
        oe=cache[key]
        logc=log_cai_for_codons(key,index)
        # Penalty for violating requested OE; still favor better OE inside threshold
        oe_loss=(cfg['oe_violation_weight']*max(0.0,oe-target_oe-cfg['oe_slack'])/2.34794
                 +cfg['oe_inside_weight']*max(0.0,oe-target_oe)/2.34794)
        loss=oe_loss+cfg['cai_weight']*(-logc)
        return loss,float(oe),float(math.exp(logc))
    current=start
    cur_loss,cur_oe,cur_cai=metrics(current)
    best=(cur_loss,cur_oe,cur_cai,current)
    for step in range(cfg['sa_steps']):
        candidate=mutate_suffix_30_to150(current,protein,rng,cfg['max_edits'])
        if candidate==current:break
        p_loss,p_oe,p_cai=metrics(candidate)
        frac=step/max(1,cfg['sa_steps']-1)
        temp=cfg['temp_start']*(cfg['temp_end']/cfg['temp_start'])**frac
        if p_loss<=cur_loss or rng.random()<math.exp(max(-700.,(cur_loss-p_loss)/max(temp,1e-10))):
            current,cur_loss,cur_oe,cur_cai=candidate,p_loss,p_oe,p_cai
        if p_loss<best[0]-1e-10 or (abs(p_loss-best[0])<=1e-10 and p_cai>best[2]):
            best=(p_loss,p_oe,p_cai,candidate)
    return best


def run_one(job):
    rec,cfg,target=job
    out=Path(cfg['output'])/'per_protein';out.mkdir(parents=True,exist_ok=True)
    path=out/f"{rec['id']}.json"
    if path.exists() and not cfg['force']:
        old=json.loads(path.read_text())
        if old.get('signature')!=cfg['signature']:
            raise RuntimeError(f"Settings mismatch for {rec['id']}; use a NEW output directory")
        return {'test_id':rec['id'],'n_local':len(old['variants']),'status':'resumed','runtime_sec':old.get('runtime_sec',0)}
    started=time.monotonic()
    index=fit_cai_index(cfg['heg_fasta'])
    seed=int(hashlib.sha256(f"{cfg['seed']}:{rec['id']}:cai_aware".encode()).hexdigest()[:12],16)
    rng=random.Random(seed)
    samples,total,exhaustive=prefix_candidates(rec,cfg['enumeration_cap'],seed,cfg['prefix_pool'])
    chosen=choose_prefixes(samples,float(target['MFE30']),index,cfg)
    cache={};variants=[]
    for pidx,(p,mfe,err,cai30) in enumerate(chosen):
        start=p+rec['dna'][30:]
        pool=[]
        for rep in range(cfg['sa_restarts']):
            loss,oe,cai150,seq=optimize_cai_aware_sa(start,rec['prot'],float(target['OE20']),index,cfg,rng,cache)
            pool.append((loss,oe,cai150,seq,rep))
        # Diverse local candidates, not identical restarts.
        unique={x[3][:150]:x for x in pool}
        if len(unique)<2:
            # Try additional local exploration, limited to explicit budget.
            for _ in range(cfg['diversity_attempts']):
                starter=rng.choice(pool)[3]
                mutated=mutate_suffix_30_to150(starter,rec['prot'],rng,2)
                if mutated[:150] in unique:continue
                # Evaluate same scores as SA, to avoid unscored rescue sequences.
                oe=cache.get(mutated[:150])
                if oe is None:oe=score_oe20(mutated,timeout=cfg['rna_timeout']);cache[mutated[:150]]=oe
                logc=log_cai_for_codons(mutated[:150],index)
                loss=(cfg['oe_violation_weight']*max(0.,oe-target['OE20']-cfg['oe_slack'])/2.34794
                      +cfg['oe_inside_weight']*max(0.,oe-target['OE20'])/2.34794
                      +cfg['cai_weight']*(-logc))
                unique[mutated[:150]]=(loss,oe,math.exp(logc),mutated,-1)
                if len(unique)>=2:break
        # Final selection also prioritizes objective + CAI, not OE20 alone.
        picked=sorted(unique.values(),key=lambda x:(x[0],-x[2],x[3]))[:2]
        for choice,x in enumerate(picked):
            loss,oe,cai150,dna,_=x
            if str(Seq(dna).translate(table=11,cds=True))!=rec['prot']:
                raise RuntimeError('Invalid protein translation for '+rec['id'])
            variants.append({'test_id':rec['id'],'prefix_index':pidx,'oe_choice':choice,
                             'dna_sequence':dna,'fixed_prefix_150nt':dna[:150],
                             'actual_MFE30':mfe,'actual_OE20':oe,
                             'CAI_prefix30':cai30,'CAI_prefix150':cai150,
                             'target_MFE30':target['MFE30'],'target_OE20':target['OE20'],
                             'local_loss':loss})
    obj={'test_id':rec['id'],'signature':cfg['signature'],'protein':rec['prot'],
         'prefix_space':total,'prefix_evaluated':len(samples),'exhaustive':exhaustive,
         'oe_evaluations':len(cache),'runtime_sec':time.monotonic()-started,'variants':variants}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False));os.replace(temp,path)
    return {'test_id':rec['id'],'n_local':len(variants),'status':'ok','runtime_sec':obj['runtime_sec']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--test-csv',type=Path,default=TEST)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--targets-json',type=Path,required=True,help='EXISTING quantiles JSON from M4_extreme_CAI08_Q98, copied unchanged')
    p.add_argument('--heg-fasta',type=Path,default=Path(DEFAULT_FASTA))
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-proteins',type=int,default=0)
    p.add_argument('--prefix-topk',type=int,default=5)
    p.add_argument('--prefix-pool',type=int,default=300)
    p.add_argument('--enumeration-cap',type=int,default=20000)
    p.add_argument('--sa-restarts',type=int,default=3)
    p.add_argument('--sa-steps',type=int,default=60)
    p.add_argument('--max-edits',type=int,default=3)
    p.add_argument('--mfe-slack',type=float,default=0.5,help='MFE gap above best achievable target error, kcal/mol')
    p.add_argument('--oe-slack',type=float,default=0.2,help='OE20 soft tolerance beyond target, kcal/mol')
    p.add_argument('--oe-violation-weight',type=float,default=5.,help='Penalty for OE20 > target+slack')
    p.add_argument('--oe-inside-weight',type=float,default=0.5)
    p.add_argument('--cai-weight',type=float,default=1.0,help='Preference for CAI_150 among good OE20 solutions')
    p.add_argument('--diversity-bonus',type=float,default=0.04)
    p.add_argument('--diversity-attempts',type=int,default=20)
    p.add_argument('--rna-timeout',type=float,default=45.)
    p.add_argument('--temp-start',type=float,default=0.5)
    p.add_argument('--temp-end',type=float,default=0.015)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    if a.workers<1 or a.sa_restarts<1 or a.sa_steps<1:raise ValueError('Invalid run settings')
    q=json.loads(a.targets_json.read_text())
    expected=('CAI','MFE30','OE20','Global_MFE_per_nt')
    if any(k not in q for k in expected):raise KeyError('Missing target in '+str(a.targets_json))
    a.output.mkdir(parents=True,exist_ok=True)
    output_targets=a.output/'00_train_quantile_targets.json'
    if output_targets.exists():
        if json.loads(output_targets.read_text()) != q:raise RuntimeError('Different targets already saved in output directory')
    else:output_targets.write_text(json.dumps(q,ensure_ascii=False,indent=2))
    model=cai_model(str(a.heg_fasta))
    print('[CAI] reference=',a.heg_fasta,'weights=',len(model),flush=True)
    print('[TARGETS]',q,flush=True)
    df=pd.read_csv(a.test_csv,dtype={'record_id':str,'dna_sequence':str,'protein_sequence':str})
    records=[]
    for _,r in df.iterrows():
        dna=str(r['dna_sequence']).strip().upper().replace('U','T')
        prot=str(r['protein_sequence']).strip().upper().rstrip('*_')
        if len(prot)<=50:continue
        if str(Seq(dna).translate(table=11,cds=True))!=prot:raise RuntimeError('Bad original reference id='+str(r['record_id']))
        records.append({'id':str(r['record_id']),'dna':dna,'prot':prot,'oe':float(r['OE20'])})
    if a.max_proteins>0:records=records[:a.max_proteins]
    signature_data={k:v for k,v in vars(a).items() if k not in ('force','max_proteins','workers')}
    signature=hashlib.sha256(json.dumps({k:str(v) for k,v in signature_data.items()},sort_keys=True).encode()).hexdigest()
    cfg={**vars(a),'output':str(a.output),'heg_fasta':str(a.heg_fasta),'signature':signature}
    tasks=[(rec,cfg,q) for rec in records]
    statuses=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for idx, result in enumerate(ex.map(run_one,tasks),1):
            statuses.append(result)
            print(f"[LOCAL] {idx}/{len(records)} id={result['test_id']} count={result['n_local']} time={result['runtime_sec']:.1f}s {result['status']}",flush=True)
    pd.DataFrame(statuses).to_csv(a.output/'01_local_audit.csv',index=False)
    rows=[]
    for rec in records:
        d=a.output/'per_protein'/f"{rec['id']}.json"
        if d.exists():rows.extend(json.loads(d.read_text())['variants'])
    if not rows:raise RuntimeError('No local sequences obtained')
    pd.DataFrame(rows).to_csv(a.output/'02_q95_local_prefixes.csv',index=False)
    print('[DONE] variants=',len(rows),'proteins=',len(records),'output=',a.output,flush=True)

if __name__=='__main__':main()
