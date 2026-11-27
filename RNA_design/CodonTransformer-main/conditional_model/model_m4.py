"""M4: fixed 50-codon prefix; two independently learned global condition branches.
Input targets use the original 4-column z-score convention for compatibility, but
ONLY columns CAI and Global_MFE are consumed. No MFE30/OE20 leakage.
"""
import json
import shutil
from pathlib import Path
import torch
from torch import nn

class FixedPrefixGlobalCodonTransformer(nn.Module):
    def __init__(self, backbone, condition_hidden=128, cai_scale=1.0, global_mfe_scale=2.0):
        super().__init__()
        self.backbone=backbone
        d=int(backbone.config.hidden_size)
        self.condition_hidden=int(condition_hidden)
        self.cai_scale=float(cai_scale)
        self.global_mfe_scale=float(global_mfe_scale)
        if self.cai_scale <= 0 or self.global_mfe_scale <= 0:
            raise ValueError('Condition scales must be positive')
        self.cai_mlp=nn.Sequential(nn.Linear(1,condition_hidden),nn.GELU(),nn.Linear(condition_hidden,d))
        self.mfe_mlp=nn.Sequential(nn.Linear(1,condition_hidden),nn.GELU(),nn.Linear(condition_hidden,d))
        self.cai_gate=nn.Parameter(torch.tensor(.01))
        self.mfe_gate=nn.Parameter(torch.tensor(.01))

    def forward(self,input_ids,attention_mask,token_type_ids,targets,labels=None):
        if targets.ndim!=2 or targets.size(1) not in (2,4):
            raise ValueError('targets must have shape [B,2] or [B,4]')
        embeddings=self.backbone.bert.embeddings.word_embeddings(input_ids)
        y=targets.to(dtype=embeddings.dtype)
        cai_h=self.cai_mlp(y[:,0:1]).unsqueeze(1)*self.cai_scale
        mfe_h=self.mfe_mlp(y[:,1:2]).unsqueeze(1)*self.global_mfe_scale
        shift=self.cai_gate.to(embeddings.dtype)*cai_h+self.mfe_gate.to(embeddings.dtype)*mfe_h
        shifted=embeddings+shift*attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return self.backbone(inputs_embeds=shifted,attention_mask=attention_mask,
                             token_type_ids=token_type_ids,labels=labels,return_dict=True)

    def save_bundle(self,path,scaler,tokenizer_path,training_args=None):
        p=Path(path);p.mkdir(parents=True,exist_ok=True)
        self.backbone.config.save_pretrained(p)
        torch.save({'architecture':'M4_fixed_prefix_v1','state_dict':self.state_dict(),
                    'condition_hidden':self.condition_hidden,
                    'cai_scale':self.cai_scale,'global_mfe_scale':self.global_mfe_scale,
                    'scaler':scaler,'training_args':training_args or {}},p/'conditional_weights.pt')
        (p/'scaler.json').write_text(json.dumps(scaler,indent=2))
        src=Path(tokenizer_path)
        if src.is_dir():
            src=src/('tokenizer.json' if (src/'tokenizer.json').exists() else 'CodonTransformerTokenizer.json')
        dst=p/'tokenizer.json'
        if src.resolve()!=dst.resolve():shutil.copyfile(src,dst)

def load_m4_bundle(path,device='cpu'):
    from transformers import BigBirdConfig,BigBirdForMaskedLM
    p=Path(path)
    ck=torch.load(p/'conditional_weights.pt',map_location='cpu',weights_only=False)
    if ck.get('architecture')!='M4_fixed_prefix_v1':raise ValueError('Not an M4 fixed-prefix checkpoint')
    backbone=BigBirdForMaskedLM(BigBirdConfig.from_pretrained(p))
    model=FixedPrefixGlobalCodonTransformer(backbone,ck['condition_hidden'],ck['cai_scale'],ck['global_mfe_scale'])
    model.load_state_dict(ck['state_dict'],strict=True)
    model.to(device).eval()
    return model,ck['scaler']
