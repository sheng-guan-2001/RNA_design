"""Fixed-prefix training masks: NEVER modify first 50 codon tokens.
Suffix positions 51..last sense codon receive supervision. Terminal STOP remains
visible and fixed to match generation. Full-mask mode uses AA-specific UNK
rather than generic [MASK], so the protein identity is visible at each position.
"""
import torch
from .collator import ConditionalMLMCollator

class FixedPrefixCollator(ConditionalMLMCollator):
    def __init__(self,vocab,prefix_codons=50,suffix_mask_rate=1.0,seed=None):
        super().__init__(vocab,mask_rate=suffix_mask_rate,full_mask_prob=0.,seed=seed)
        if prefix_codons<1 or not 0<suffix_mask_rate<=1:
            raise ValueError('prefix_codons>=1 and suffix_mask_rate in (0,1] required')
        self.prefix_codons=int(prefix_codons)
        self.suffix_mask_rate=float(suffix_mask_rate)

    def __call__(self,examples):
        B=len(examples);L=max(len(x['input_ids']) for x in examples)
        ids=torch.full((B,L),self.pad,dtype=torch.long)
        attn=torch.zeros_like(ids)
        hosts=torch.zeros_like(ids)
        labels=torch.full_like(ids,-100)
        targets=torch.tensor([x['targets'] for x in examples],dtype=torch.float32)
        for i,ex in enumerate(examples):
            row=ex['input_ids'];n=len(row)
            # [CLS], sense_codons..., STOP, [SEP] => n-3 sense codons
            if n-3<=self.prefix_codons:
                raise ValueError(f'No suffix after {self.prefix_codons} codons: n={n}')
            ids[i,:n]=torch.tensor(row,dtype=torch.long)
            attn[i,:n]=1
            hosts[i,:n]=ex['organism']
            positions=list(range(self.prefix_codons+1,n-2))
            selected=[p for p in positions if self._rand()<self.suffix_mask_rate]
            if not selected:selected=[positions[int(self._rand()*len(positions))]]
            for p in selected:
                old=int(ids[i,p]);labels[i,p]=old
                # Always use AA-specific mask, including mask rate=1.0.
                # Standard MLM 80/10/10 would leak original codon in some supervised positions.
                ids[i,p]=self.unk_map[old]
            assert torch.all(labels[i,1:self.prefix_codons+1]==-100)
            assert torch.all(ids[i,1:self.prefix_codons+1]==torch.tensor(row[1:self.prefix_codons+1]))
            assert labels[i,n-2].item()==-100  # STOP always visible
        return {'input_ids':ids,'attention_mask':attn,'token_type_ids':hosts,
                'targets':targets,'labels':labels}
