"""Generate CDS suffix conditioned on CAI and Global_MFE, LOCK first 150nt.
Input: two_stage_100/02_best_sequences.csv and reserved original test CSV.
Run: python -m conditional_model.generate_m4 --help
"""
import argparse,csv,json,random
from pathlib import Path
import torch
from Bio.Seq import Seq
from Bio.Data import CodonTable
from .config import ROOT,MAX_LEN
from .dataset import load_vocab
from .model_m4 import load_m4_bundle

DEFAULT_TEST=ROOT/'14_reserved_test_four_objectives.csv'

def parse():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--optimized-csv',type=Path,required=True,help='02_best_sequences.csv from local search')
 p.add_argument('--reference-csv',type=Path,default=DEFAULT_TEST)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--method',default='SA')
 p.add_argument('--prefix-codons',type=int,default=50)
 p.add_argument('--organism-id',type=int,default=51)
 p.add_argument('--n-proteins',type=int,default=0)
 p.add_argument('--replicates',type=int,default=2)
 p.add_argument('--temperature',type=float,default=1.0)
 p.add_argument('--gpu-batch-size',type=int,default=4)
 p.add_argument('--delta-z',type=float,default=.75)
 p.add_argument('--max-abs-z',type=float,default=2.5)
 p.add_argument('--global-mfe-relative-cap',type=float,default=.2)
 p.add_argument('--seed',type=int,default=2026)
 p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
 return p.parse_args()

def allowed_dict(vocab):
 table=CodonTable.unambiguous_dna_by_id[11]
 out={}
 for aa in 'ACDEFGHIKLMNPQRSTVWY':
  codons=[c for c,a in table.forward_table.items() if a==aa]
  if aa=='M':codons=['ATG']
  tok=[vocab[f'{aa.lower()}_{c.lower()}'] for c in sorted(codons) if f'{aa.lower()}_{c.lower()}' in vocab]
  if not tok:raise ValueError(f'No codon tokens for {aa}')
  out[aa]=tok
  if f'{aa.lower()}_unk' not in vocab:raise ValueError(f'Missing {aa.lower()}_unk')
 return out

def records(ref_file):
 import pandas as pd
 df=pd.read_csv(ref_file,dtype={'record_id':str,'dna_sequence':str,'protein_sequence':str})
 out={}
 for _,r in df.iterrows():
  t=str(r['record_id']);dna=str(r['dna_sequence']).strip().upper().replace('U','T')
  protein=str(r['protein_sequence']).strip().upper().rstrip('*_')
  if t in out:raise ValueError(f'Duplicate reference record_id: {t}')
  if str(Seq(dna).translate(table=11,cds=True))!=protein:raise ValueError(f'Invalid reference {t}')
  out[t]={'dna':dna,'protein':protein,'CAI':float(r['CAI']),'Global_MFE':float(r['Global_MFE'])}
 return out

def scenarios(ref,scaler,a):
 names=['matched','CAI_high','CAI_low','Global_MFE_high','Global_MFE_low']
 result=[]
 for name in names:
  raw={'CAI':ref['CAI'],'Global_MFE':ref['Global_MFE']}
  if name!='matched':
   key,side=name.rsplit('_',1)
   sign=1 if side=='high' else -1
   span=a.delta_z*scaler[key]['std']
   if key=='Global_MFE':span=min(span,abs(ref[key])*a.global_mfe_relative_cap)
   raw[key]+=sign*span
   if key=='CAI':raw[key]=min(1.,max(0.,raw[key]))
   if key=='Global_MFE':raw[key]=min(0.,raw[key])
  z={key:(raw[key]-scaler[key]['mean'])/scaler[key]['std'] for key in raw}
  # Limit z to same safe training-centered range. Preserve actual corresponding raw target.
  for key in z:
   z[key]=min(a.max_abs_z,max(-a.max_abs_z,z[key]))
   raw[key]=scaler[key]['mean']+z[key]*scaler[key]['std']
  result.append((name,raw,z))
 return result

@torch.inference_mode()
def generate_batch(model,vocab,inv,allowed,protein,prefix_dna,stop,
                   cond_rows,seeds,a,device):
 prefix_codons=[prefix_dna[j:j+3] for j in range(0,len(prefix_dna),3)]
 # [CLS] + protein sense codons + STOP + [SEP]; prefix 1..50 preserved.
 amino=protein
 B=len(cond_rows)
 n=len(amino)+3
 inp=torch.empty((B,n),dtype=torch.long,device=device)
 inp[:,0]=vocab['[CLS]'];inp[:,-1]=vocab['[SEP]']
 inp[:,-2]=vocab[f'__{stop.lower()}']
 for j,aa in enumerate(amino):
  if j<len(prefix_codons):
   c=prefix_codons[j]
   # For bacterial alternative initiators, original first-codon token can use its own first-AA
   # annotation; tokenizer may have m_gtg/m_ttg or v_gtg/l_ttg respectively.
   possibilities=[f'{aa.lower()}_{c.lower()}']
   if j==0 and aa=='M':
    from Bio.Data import CodonTable
    true_aa=CodonTable.unambiguous_dna_by_id[11].forward_table.get(c)
    if true_aa:possibilities.append(f'{true_aa.lower()}_{c.lower()}')
   toks=[vocab[k] for k in possibilities if k in vocab]
   if not toks:raise ValueError(f'Fixed-prefix codon absent from vocabulary at {j}: {aa} {c}')
   inp[:,j+1]=toks[0]
  else:inp[:,j+1]=vocab[f'{aa.lower()}_unk']
 target=torch.tensor([[z['CAI'],z['Global_MFE']] for _,_,z in cond_rows],device=device,dtype=torch.float32)
 logits=model(input_ids=inp,attention_mask=torch.ones_like(inp),
              token_type_ids=torch.full_like(inp,a.organism_id),targets=target).logits
 # Vectorized synonymous-only sampling over the whole suffix; one GPU draw per candidate.
 suffix_amino=amino[len(prefix_codons):]
 K=max(len(allowed[aa]) for aa in suffix_amino)
 choices=torch.zeros((len(suffix_amino),K),dtype=torch.long,device=device)
 legal=torch.zeros_like(choices,dtype=torch.bool)
 for j,aa in enumerate(suffix_amino):
  idx=allowed[aa]
  choices[j,:len(idx)]=torch.tensor(idx,dtype=torch.long,device=device)
  legal[j,:len(idx)]=True
 positions=1+len(prefix_codons)+torch.arange(len(suffix_amino),device=device)
 chosen_logits=logits[:,positions,:].gather(-1,choices[None,:,:].expand(B,-1,-1)).float()/a.temperature
 chosen_logits=chosen_logits.masked_fill(~legal[None,:,:],-1e9)
 probs=torch.softmax(chosen_logits,dim=-1)
 suffix_codons=[]
 for b in range(B):
  rng=torch.Generator(device=device).manual_seed(seeds[b])
  sampled=torch.multinomial(probs[b],num_samples=1,generator=rng)
  selected=choices.gather(1,sampled).flatten().cpu().tolist()
  suffix_codons.append([inv[t].rsplit('_',1)[1].upper() for t in selected])
 out=[]
 for codons in suffix_codons:
  dna=prefix_dna+''.join(codons)+stop
  if dna[:len(prefix_dna)]!=prefix_dna:raise RuntimeError('Prefix changed')
  if str(Seq(dna).translate(table=11,cds=True))!=protein:raise RuntimeError('Synonymous translation changed')
  out.append(dna)
 return out

def main():
 a=parse()
 if a.temperature<=0 or a.replicates<1 or a.gpu_batch_size<1:raise ValueError('Bad sample settings')
 device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(4)
 model,scaler=load_m4_bundle(a.checkpoint,device)
 train_prefix=int(torch.load(a.checkpoint/'conditional_weights.pt',map_location='cpu',weights_only=False)['training_args'].get('prefix_codons',50))
 if a.prefix_codons!=train_prefix:raise ValueError(f'Prefix mismatch: trained={train_prefix}, requested={a.prefix_codons}')
 vocab=load_vocab(a.checkpoint);inv={v:k for k,v in vocab.items()};allowed=allowed_dict(vocab)
 ref=records(a.reference_csv)
 import pandas as pd
 df=pd.read_csv(a.optimized_csv,dtype={'test_id':str,'dna_sequence':str})
 if a.method:df=df[df['method']==a.method]
 if df.empty:raise ValueError('No matching optimized sequences')
 # user may have both mfe_high__oe_low / mfe_low__oe_high for same protein
 rows=df.to_dict('records')
 rows=sorted(rows,key=lambda x:(str(x['test_id']),str(x.get('scenario',''))))
 seen=set();eligible=[]
 for row in rows:
  rid=str(row['test_id']);label=str(row.get('scenario',''))
  key=(rid,label)
  if key in seen:raise ValueError(f'Duplicate optimized entry: {key}')
  seen.add(key)
  if rid not in ref:raise ValueError(f'Unknown test_id in reference CSV: {rid}')
  protein=ref[rid]['protein'];dna=str(row['dna_sequence']).upper().replace('U','T')
  if str(Seq(dna).translate(table=11,cds=True))!=protein:raise ValueError(f'Bad optimized CDS: {key}')
  if len(protein)<=a.prefix_codons:
   print(f'[SKIP] {rid} length={len(protein)} <= prefix={a.prefix_codons}',flush=True)
   continue
  if len(protein)+3>MAX_LEN:raise ValueError(f'Too long {rid}')
  eligible.append((rid,label,dna,protein,ref[rid]))
 if a.n_proteins>0:
  first_ids=list(dict.fromkeys(row[0] for row in eligible))[:a.n_proteins]
  eligible=[item for item in eligible if item[0] in first_ids]
 if not eligible:raise ValueError('No eligible proteins')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 fields=['test_id','local_scenario','global_scenario','replicate','protein_sequence',
         'fixed_prefix_150nt','generated_dna','reference_CAI','reference_Global_MFE',
         'target_CAI','target_Global_MFE','target_z_CAI','target_z_Global_MFE',
         'local_MFE30','local_OE20','synonymous_valid']
 total=0
 with a.output.open('w',newline='',encoding='utf8') as fh:
  w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
  for i,(rid,local_scenario,dna,protein,reference) in enumerate(eligible):
   prefix=dna[:a.prefix_codons*3];stop=dna[-3:]
   pairs=[]
   for scen,raw,z in scenarios(reference,scaler,a):
    for rep in range(a.replicates):pairs.append((scen,raw,z,rep))
   for start in range(0,len(pairs),a.gpu_batch_size):
    group=pairs[start:start+a.gpu_batch_size]
    cond_rows=[(sc,raw,z) for sc,raw,z,_ in group]
    # stable seeds across conditions in same local prefix
    rng=[a.seed+i*100003+rep*997 for _,_,_,rep in group]
    dna_rows=generate_batch(model,vocab,inv,allowed,protein,prefix,stop,cond_rows,rng,a,device)
    for (sc,raw,z,rep),final_dna in zip(group,dna_rows):
     w.writerow({'test_id':rid,'local_scenario':local_scenario,'global_scenario':sc,
        'replicate':rep,'protein_sequence':protein,'fixed_prefix_150nt':prefix,'generated_dna':final_dna,
        'reference_CAI':reference['CAI'],'reference_Global_MFE':reference['Global_MFE'],
        'target_CAI':raw['CAI'],'target_Global_MFE':raw['Global_MFE'],
        'target_z_CAI':z['CAI'],'target_z_Global_MFE':z['Global_MFE'],
        'local_MFE30':'','local_OE20':'',
        'synonymous_valid':True})
     total+=1
   fh.flush()
   if (i+1)%10==0 or i+1==len(eligible):print(f'[GEN] {i+1}/{len(eligible)} prefixes, {total} CDS',flush=True)
 manifest={'checkpoint':str(a.checkpoint),'optimized_csv':str(a.optimized_csv),
           'reference_csv':str(a.reference_csv),'n_prefixes':len(eligible),'n_generated':total,
           'prefix_codons':a.prefix_codons,'sampling':'single_forward_all_suffix_masked',
           'note':'RNA metrics must be calculated independently on generated_dna'}
 a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
 print(f'[DONE] {total} CDS -> {a.output}',flush=True)
if __name__=='__main__':main()
