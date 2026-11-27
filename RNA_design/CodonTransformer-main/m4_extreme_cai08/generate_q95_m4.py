#!/usr/bin/env python3
"""Use fixed Q80/Q20 goals to generate 2 CDS for each of 10 local prefixes.
Model is the original trained M4 checkpoint; no retraining or score optimization.
"""
import argparse,csv,json,math,random,hashlib
from pathlib import Path
import pandas as pd
import torch
from Bio.Seq import Seq
from conditional_model.dataset import load_vocab
from conditional_model.model_m4 import load_m4_bundle
from conditional_model.generate_m4 import allowed_dict, generate_batch, records
from conditional_model.config import MAX_LEN

BASE=Path('/mnt/sdc/guansheng/protein_RNA/clear_understand')
OUT=BASE/'model_training/M4_q95_joint20'
CKPT=BASE/'model_training/conditional_m4_fixed_prefix_fullmask/epochs/epoch_10'
TEST=BASE/'data_preparation/ecoli_codontransformer/14_reserved_test_four_objectives.csv'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--local-csv',type=Path,default=OUT/'02_q95_local_prefixes.csv')
    p.add_argument('--quantiles',type=Path,default=OUT/'00_train_quantile_targets.json')
    p.add_argument('--reference-csv',type=Path,default=TEST)
    p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--output',type=Path,default=OUT/'03_m4_q95_generated.csv')
    p.add_argument('--replicates',type=int,default=2)
    p.add_argument('--gpu-batch-size',type=int,default=4)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--max-abs-z',type=float,default=2.5)
    p.add_argument('--prefix-codons',type=int,default=50)
    p.add_argument('--organism-id',type=int,default=51)
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='auto')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-proteins',type=int,default=0)
    a=p.parse_args()
    if a.replicates<1 or a.gpu_batch_size<1 or a.temperature<=0:raise ValueError('Invalid sampling settings')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    q=json.loads(a.quantiles.read_text()); refs=records(a.reference_csv)
    print('[TARGETS]',json.dumps(q,ensure_ascii=False),flush=True)
    local=pd.read_csv(a.local_csv,dtype={'test_id':str,'dna_sequence':str})
    ids=list(dict.fromkeys(local['test_id'].tolist()))
    if a.max_proteins>0:ids=ids[:a.max_proteins]
    local=local[local.test_id.isin(ids)].sort_values(['test_id','prefix_index','oe_choice'])
    model,scaler=load_m4_bundle(a.checkpoint,device)
    ck=torch.load(a.checkpoint/'conditional_weights.pt',map_location='cpu',weights_only=False)
    trained_prefix=int(ck['training_args'].get('prefix_codons',50))
    if trained_prefix!=a.prefix_codons:raise ValueError('Mismatch trained prefix length')
    vocab=load_vocab(a.checkpoint);inv={v:k for k,v in vocab.items()};allowed=allowed_dict(vocab)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=['test_id','local_scenario','global_scenario','replicate','candidate_rank','prefix_index','oe_choice',
            'protein_sequence','fixed_prefix_150nt','generated_dna','reference_CAI','reference_Global_MFE',
            'target_CAI','target_Global_MFE','requested_CAI','requested_Global_MFE',
            'target_z_CAI','target_z_Global_MFE','target_clipped_CAI','target_clipped_Global_MFE',
            'local_MFE30','local_OE20','target_MFE30','target_OE20','synonymous_valid']
    rows=[]
    # Reproducible random seeds per (test_id, prefix_index, oe_choice, replicate)
    for k,(rid,group) in enumerate(local.groupby('test_id',sort=False),1):
        if rid not in refs:raise KeyError('Missing protein reference: '+rid)
        ref=refs[rid];protein=ref['protein'];stop=ref['dna'][-3:]
        if len(protein)+3>MAX_LEN:raise ValueError('Sequence too long: '+rid)
        requested={'CAI':float(q['CAI']),
                   'Global_MFE':float(q['Global_MFE_per_nt'])*len(ref['dna'])}
        z={};raw={};clipped={}
        for key in ('CAI','Global_MFE'):
            x=(requested[key]-scaler[key]['mean'])/scaler[key]['std']
            z[key]=max(-a.max_abs_z,min(a.max_abs_z,x))
            raw[key]=scaler[key]['mean']+z[key]*scaler[key]['std']
            clipped[key]=abs(x-z[key])>1e-8
        print(f"[GOAL] id={rid} requested_CAI={requested['CAI']:.4f} used_CAI={raw['CAI']:.4f} requested_G={requested['Global_MFE']:.2f} used_G={raw['Global_MFE']:.2f} clipped={clipped}",flush=True)
        # Same CAI / Global MFE conditions across all 20 candidates of this protein
        for _,loc in group.iterrows():
            start=str(loc['dna_sequence']).upper().replace('U','T')
            if str(Seq(start).translate(table=11,cds=True))!=protein:raise ValueError(f'Invalid local {rid}')
            prefix=start[:150]
            cond_rows=[('Q95_joint',raw,z)]*a.replicates
            seeds=[]
            for rep in range(a.replicates):
                key=f"{a.seed}:{rid}:{int(loc['prefix_index'])}:{int(loc['oe_choice'])}:{rep}"
                seeds.append(int(hashlib.sha256(key.encode()).hexdigest()[:12],16)%(2**63-1))
            generated=[]
            for start_idx in range(0,len(cond_rows),a.gpu_batch_size):
                generated.extend(generate_batch(model,vocab,inv,allowed,protein,prefix,stop,
                              cond_rows[start_idx:start_idx+a.gpu_batch_size],
                              seeds[start_idx:start_idx+a.gpu_batch_size],a,device))
            for rep,dna in enumerate(generated):
                rows.append({'test_id':rid,'local_scenario':'mfe_q95__oe_q05',
                             'global_scenario':'Q95_joint','replicate':rep,
                             'candidate_rank':len([r for r in rows if r['test_id']==rid])+1,
                             'prefix_index':int(loc['prefix_index']),'oe_choice':int(loc['oe_choice']),
                             'protein_sequence':protein,'fixed_prefix_150nt':prefix,'generated_dna':dna,
                             'reference_CAI':ref['CAI'],'reference_Global_MFE':ref['Global_MFE'],
                             'target_CAI':raw['CAI'],'target_Global_MFE':raw['Global_MFE'],
                             'requested_CAI':requested['CAI'],'requested_Global_MFE':requested['Global_MFE'],
                             'target_z_CAI':z['CAI'],'target_z_Global_MFE':z['Global_MFE'],
                             'target_clipped_CAI':clipped['CAI'],'target_clipped_Global_MFE':clipped['Global_MFE'],
                             'local_MFE30':loc['actual_MFE30'],'local_OE20':loc['actual_OE20'],
                             'target_MFE30':q['MFE30'],'target_OE20':q['OE20'],'synonymous_valid':True})
        g=[x for x in rows if x['test_id']==rid]
        print(f"[GENERATED] {k}/{len(ids)} id={rid}: total={len(g)} unique={len({x['generated_dna'] for x in g})}",flush=True)
    df=pd.DataFrame(rows,columns=fields)
    df.to_csv(a.output,index=False)
    audit=df.groupby('test_id').agg(n=('generated_dna','size'),unique=('generated_dna','nunique'),
              cai_clipped=('target_clipped_CAI','max'),g_clipped=('target_clipped_Global_MFE','max')).reset_index()
    audit.to_csv(a.output.with_name('04_generation_audit.csv'),index=False)
    manifest={'checkpoint':str(a.checkpoint),'quantiles':q,'n_sequences':len(df),'n_proteins':len(ids),
              'prefixes_per_protein':10,'replicates_per_prefix':a.replicates,
              'interpretation':'Training quantiles with explicitly documented manual overrides',
              'note':'Separate explicit target clipping flags. Local search and evaluation are computational proxies.'}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('[DONE]',str(a.output),flush=True)

if __name__=='__main__':main()
