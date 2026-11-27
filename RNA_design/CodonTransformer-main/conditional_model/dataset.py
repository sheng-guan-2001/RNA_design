"""Read EXACT prepare_ecoli_codontransformer JSONL schema; never recompute z-score on validation."""
import json
import math
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .config import FEATURES, MAX_LEN


def load_scaler(path):
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    for k in FEATURES:
        if k not in obj or not math.isfinite(float(obj[k]['mean'])) or not math.isfinite(float(obj[k]['std'])) or float(obj[k]['std']) <= 0:
            raise ValueError(f'Invalid scaler entry for {k}: {obj.get(k)}')
    return {k: {'mean':float(obj[k]['mean']), 'std':float(obj[k]['std'])} for k in FEATURES}


def prepare_item(obj, scaler, vocab, max_len=MAX_LEN, organism_id=51):
    if int(obj['organism']) != organism_id:
        raise ValueError(f"Unexpected organism {obj['organism']}, expected {organism_id}")
    s = str(obj['codons']).strip().lower().split()
    if len(s) < 2 or len(s)+2 > max_len:
        raise ValueError(f'CDS tokens exceed model capacity or too short: {len(s)}')
    if s[-1] not in ('__taa', '__tag', '__tga'):
        raise ValueError(f'Missing terminal stop token: {s[-1]}')
    if any(x not in vocab or vocab[x] < 26 for x in s):
        raise ValueError('Unknown/non-codon token in: '+str(s[:5]))
    raw = obj.get('targets')
    if not isinstance(raw, dict):
        raise ValueError('JSONL requires targets mapping')
    z = [(float(raw[k])-scaler[k]['mean']) / scaler[k]['std'] for k in FEATURES]
    if not all(math.isfinite(v) for v in z):
        raise ValueError('Nonfinite target values')
    stored = obj.get('targets_z')
    if stored is not None:
        for i,k in enumerate(FEATURES):
            if not math.isclose(float(stored[k]),z[i],rel_tol=1e-5,abs_tol=1e-4):
                raise ValueError(f'targets_z mismatched actual train-only scaler for {k}')
    if 'dna_sequence' in obj:
        dna=''.join(x.rsplit('_',1)[1].upper() for x in s)
        if dna != str(obj['dna_sequence']).strip().upper().replace('U','T'):
            raise ValueError('codons and dna_sequence disagree')
    # Token IDs are defined in EXACT original vocabulary; no subword fallback.
    ids = [vocab['[CLS]']] + [vocab[x] for x in s] + [vocab['[SEP]']]
    return {'input_ids':ids, 'targets':z, 'organism':organism_id,
            'protein':str(obj.get('protein_sequence','')), 'raw_targets': [float(raw[k]) for k in FEATURES]}


class ConditionalCDSDataset(Dataset):
    def __init__(self, path, scaler, vocab, max_len=MAX_LEN, organism_id=51, limit=0):
        self.items=[]
        with open(path,encoding='utf-8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                try: item=prepare_item(json.loads(line),scaler,vocab,max_len,organism_id)
                except Exception as e: raise ValueError(f'{path}:{line_no}: {e}') from e
                self.items.append(item)
                if limit and len(self.items)>=limit:break
        if not self.items: raise ValueError(f'No records: {path}')
        print(f'[DATA] {path}: {len(self.items):,} CDS',flush=True)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def load_vocab(tokenizer_path):
    """Load exact existing Tokenizers vocabulary. tokenizer.json is sufficient."""
    p=Path(tokenizer_path)
    if p.is_dir():
        p = (p/'tokenizer.json') if (p/'tokenizer.json').exists() else (p/'CodonTransformerTokenizer.json')
    if not p.exists():raise FileNotFoundError(f'tokenizer file missing: {p}')
    with open(p,encoding='utf-8') as f: data=json.load(f)
    vocab=data['model']['vocab']
    need=['[CLS]','[SEP]','[PAD]','__taa','__tag','__tga']
    for k in need:
        if k not in vocab:raise ValueError(f'Tokenizer missing {k}')
    ids=list(vocab.values())
    if sorted(ids)!=list(range(len(ids))):raise ValueError('Tokenizer IDs are not contiguous')
    return vocab
