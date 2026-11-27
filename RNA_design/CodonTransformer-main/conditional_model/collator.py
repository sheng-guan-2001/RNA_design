"""Amino-acid-preserving masked codon training. No forbidden substitutions."""
import torch

class ConditionalMLMCollator:
    def __init__(self, vocab, mask_rate=.35, full_mask_prob=.15, seed=None):
        self.vocab=vocab
        self.mask_rate=mask_rate
        self.full_mask_prob=full_mask_prob
        if not 0 <= mask_rate <= 1 or not 0 <= full_mask_prob <= 1:
            raise ValueError('Probabilities must be 0..1')
        self.pad=vocab['[PAD]']; self.cls=vocab['[CLS]']; self.sep=vocab['[SEP]']
        self.unk_map={}
        self.by_aa={}
        for name,idx in vocab.items():
            if idx < 26 or '_' not in name:continue
            aa,codon=name.rsplit('_',1)
            if len(codon)!=3 or set(codon)-set('acgt'):continue
            unk=f'{aa}_unk'
            if unk not in vocab:raise ValueError(f'Missing mask token: {unk}')
            self.unk_map[idx]=vocab[unk]
            self.by_aa.setdefault(aa,[]).append(idx)
        self.generator=None
        if seed is not None:
            self.generator=torch.Generator().manual_seed(seed)

    def _rand(self):
        return torch.rand((),generator=self.generator).item()

    def __call__(self,examples):
        b=len(examples); length=max(len(x['input_ids']) for x in examples)
        ids=torch.full((b,length), self.pad,dtype=torch.long)
        attn=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
        hosts=torch.zeros_like(ids)
        y=torch.tensor([x['targets'] for x in examples],dtype=torch.float32)
        for i,ex in enumerate(examples):
            row=ex['input_ids']; n=len(row)
            ids[i,:n]=torch.tensor(row,dtype=torch.long)
            attn[i,:n]=1;hosts[i,:n]=ex['organism']
            # Do not mask special tokens; full_mask applies to ALL codons including STOP.
            positions=list(range(1,n-1))
            full = self._rand() < self.full_mask_prob
            chosen=[p for p in positions if full or self._rand()<self.mask_rate]
            if not chosen:chosen=[positions[int(self._rand()*len(positions))]]
            for p in chosen:
                original=int(ids[i,p]); labels[i,p]=original
                r=self._rand()
                if r < 0.8: ids[i,p]=self.unk_map[original]
                elif r < 0.9:
                    aa=next(k for k,v in self.by_aa.items() if original in v)
                    possibilities=self.by_aa[aa]
                    ids[i,p]=possibilities[int(self._rand()*len(possibilities))]
                # else leave original unchanged
        return dict(input_ids=ids,attention_mask=attn,token_type_ids=hosts,
                    labels=labels,targets=y)
