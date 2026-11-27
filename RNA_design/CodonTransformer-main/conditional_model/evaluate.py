"""Recompute objectives from generated CDS using same CDS-only definitions as data preparation.
Requires ViennaRNA RNA Python bindings and RNAplfold executable, plus HEG FASTA.
"""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils import CodonAdaptationIndex
import RNA
from .config import FEATURES

R=0.00198720425864083
DEFAULT_FASTA='/mnt/sdc/guansheng/protein_RNA/dataset/ecol.heg.fasta/ecol.heg.fasta'

def cai_model(fasta):
    refs=[]
    for r in SeqIO.parse(str(fasta),'fasta'):
        s=str(r.seq).upper().replace('U','T')
        if s and len(s)%3==0 and not set(s)-set('ACGT'):refs.append(s)
    if len(refs)<10:raise ValueError('Too few CAI reference CDS')
    return CodonAdaptationIndex(refs)

def opening(seq, temperature=37, window=150, span=100):
    if len(seq)<30:raise ValueError('CDS requires at least 30 nt')
    with tempfile.TemporaryDirectory() as work:
        command=['RNAplfold','-W',str(window),'-L',str(span),'-u','30','-T',str(temperature)]
        proc=subprocess.run(command,input='>cds\n'+seq+'\n',text=True,capture_output=True,cwd=work,timeout=600)
        if proc.returncode:raise RuntimeError(f'RNAplfold failed: {proc.stderr[-1000:]}')
        paths=list(Path(work).glob('*_lunp'))
        if len(paths)!=1:raise RuntimeError(f'Expected one lunp, got {paths}')
        for line in paths[0].read_text().splitlines():
            parts=line.split()
            if parts and parts[0]=='20':
                p=float(parts[20])
                if not 0<p<=1:raise ValueError(f'Invalid Popen20={p}')
                return -R*(273.15+temperature)*math.log(p)
        raise ValueError('RNAplfold has no end position 20')

def metrics(dna,cai,temperature=37,window=150,span=100):
    dna=dna.upper().replace('U','T');rna=dna.replace('T','U')
    RNA.cvar.temperature=float(temperature)
    return {'CAI':float(cai.calculate(dna)),
            'Global_MFE':float(RNA.fold(rna)[1]),
            'MFE30':float(RNA.fold(rna[:30])[1]),
            'OE20':float(opening(rna,temperature,window,span))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--heg-fasta',type=Path,default=DEFAULT_FASTA)
    p.add_argument('--temperature',type=float,default=37.0)
    p.add_argument('--window',type=int,default=150)
    p.add_argument('--span',type=int,default=100)
    args=p.parse_args()
    data=json.loads(args.generated.read_text())
    actual=metrics(data['dna_sequence'],cai_model(args.heg_fasta),args.temperature,args.window,args.span)
    print(json.dumps({'target':data['targets'],'actual':actual,
                      'absolute_error':{k:abs(data['targets'][k]-actual[k]) for k in FEATURES}},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
