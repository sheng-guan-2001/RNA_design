#!/usr/bin/env python3
"""Audit ColiFormer CAI.csv, finetune_set.json (JSONL), test_set.json (JSON array/JSONL).
No RNA folding is performed. CAI.csv genome provenance is NOT automatically E. coli.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Bio.Data import CodonTable
from Bio.Seq import Seq

TABLE = CodonTable.unambiguous_dna_by_id[11]
STOP = set(TABLE.stop_codons)
CODON = re.compile(r'^[ACGT]{3}$')
ACC = re.compile(r'^(NC_\d+\.\d+)_cds_')


def check_cds(seq, tagged_protein=None):
    seq = str(seq).strip().upper().replace('U', 'T')
    if not seq or len(seq) % 3:
        return None, 'invalid_length'
    if not re.fullmatch('[ACGT]+', seq):
        return None, 'invalid_nucleotide'
    triplets = [seq[k:k + 3] for k in range(0, len(seq), 3)]
    if any(t in STOP for t in triplets[:-1]):
        return None, 'internal_stop'
    has_stop = triplets[-1] in STOP
    coding = seq[:-3] if has_stop else seq
    protein = str(Seq(coding).translate(table=11))
    if not protein or '*' in protein:
        return None, 'translation_failure'
    # Bacterial translation starts can be alternative start codons.
    if triplets[0] in TABLE.start_codons:
        protein = 'M' + protein[1:]
    if tagged_protein is not None and protein != tagged_protein:
        return None, 'token_amino_acid_mismatch'
    return (seq, protein, has_stop), 'valid'


def read_json_records(path):
    # Supports both JSON arrays and one-JSON-object-per-line (JSONL).
    with path.open('r', encoding='utf-8-sig') as f:
        first = ''
        while True:
            c = f.read(1)
            if not c: break
            if not c.isspace():
                first = c
                break
    if first == '[':
        with path.open('r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f'{path}: JSON root must be a list')
        for i, item in enumerate(data, 1):
            yield i, item
    else:
        with path.open('r', encoding='utf-8-sig') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield i, json.loads(line)
                    except json.JSONDecodeError:
                        yield i, {'_parse_error': True}


def decode_tokens(s):
    """CodonTransformer-style tokens A_GCT, __TAA."""
    parts = str(s).split()
    if not parts: return None, None, 'empty_tokens'
    codons, amino = [], []
    for ix, token in enumerate(parts):
        if '_' not in token:
            return None, None, 'malformed_token'
        a, c = token.rsplit('_', 1)
        a, c = a.upper(), c.upper()
        if not CODON.fullmatch(c):
            return None, None, 'invalid_token_codon'
        if a in ('', '_', '*', 'STOP'):
            if c not in STOP or ix != len(parts) - 1:
                return None, None, 'invalid_stop_token'
        elif len(a) == 1 and a.isalpha():
            translated = str(Seq(c).translate(table=11))
            if ix == 0 and c in TABLE.start_codons:
                translated = 'M'
            if translated != a:
                return None, None, 'token_amino_acid_mismatch'
            amino.append(a)
        else:
            return None, None, 'malformed_amino_token'
        codons.append(c)
    return ''.join(codons), ''.join(amino), 'valid'


def parse_cai(path, records, stats, errors, accession):
    # Typical columns: Unnamed: 0, CAI, Unnamed: 2, Unnamed: 3, Sequence, #N/A
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        dna_key = next((c for c in columns if c.strip().lower() in ('sequence','dna_sequence','dna','cds')), None)
        if dna_key is None:
            raise ValueError(f'No DNA sequence column in {path}; columns={columns}')
        id_key = columns[0]
        for lineno, row in enumerate(reader, 2):
            stats['CAI']['total'] += 1
            seq = row.get(dna_key, '')
            obj, status = check_cds(seq)
            if obj is None:
                errors['CAI'][status] += 1
                continue
            dna, protein, stop = obj
            rawid = (row.get(id_key) or '').strip()
            acc = ACC.match(rawid)
            acc = acc.group(1) if acc else 'UNKNOWN'
            accession[acc] += 1
            cai = row.get('CAI')
            records.append(dict(source='CAI', record_id=rawid or f'line_{lineno}', dna=dna,
                                protein=protein, terminal_stop=stop, organism='unverified',
                                accession=acc, provided_cai=cai))
            stats['CAI']['valid'] += 1


def parse_json(path, name, records, stats, errors):
    for lineno, item in read_json_records(path):
        stats[name]['total'] += 1
        if not isinstance(item, dict) or item.get('_parse_error'):
            errors[name]['bad_json_record'] += 1
            continue
        if str(item.get('organism', '')) != '51':
            errors[name]['organism_not_51'] += 1
            continue
        raw = item.get('codons', '')
        if isinstance(raw, str) and '_' in raw and ' ' in raw:
            dna, token_protein, status = decode_tokens(raw)
            if status != 'valid':
                errors[name][status] += 1
                continue
        else:
            dna, token_protein = str(raw).strip(), None
        obj, status = check_cds(dna, tagged_protein=token_protein)
        if obj is None:
            errors[name][status] += 1
            continue
        dna, protein, stop = obj
        records.append(dict(source=name, record_id=item.get('idx', lineno), dna=dna,
                            protein=protein, terminal_stop=stop, organism='51',
                            accession='', provided_cai=''))
        stats[name]['valid'] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='/mnt/sdc/guansheng/protein_RNA/clear_understand/dataset/ColiFormer-Data')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--allow-unverified-cai', action='store_true',
                    help='Include CAI.csv in candidate training set even though source organisms are unverified (not recommended)')
    args = ap.parse_args()
    root = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else root/'audit_results'
    out.mkdir(parents=True, exist_ok=True)
    stats, errors, acc = defaultdict(Counter), defaultdict(Counter), Counter()
    records = []
    paths = {'CAI':root/'CAI.csv','finetune':root/'finetune_set.json','test':root/'test_set.json'}
    for name,p in paths.items():
        if not p.exists():
            print(f'[MISSING] {p}')
            continue
        print(f'[READ] {p}', flush=True)
        if name == 'CAI': parse_cai(p, records, stats, errors, acc)
        else: parse_json(p, name, records, stats, errors)

    # Normalizing terminal stop does not conflate different stop-codon identities;
    # it allows exact CDS duplicate checks independent of terminal stop presence.
    def coding_key(r):
        return r['dna'][:-3] if r['terminal_stop'] else r['dna']
    for r in records: r['coding_dna'] = coding_key(r)

    by_source = {name: [r for r in records if r['source']==name] for name in paths}
    keys = {name: {r['coding_dna'] for r in rows} for name,rows in by_source.items()}
    proteins = {name: {r['protein'] for r in rows} for name,rows in by_source.items()}
    test_keys = keys['test']
    test_proteins = proteins['test']

    overview = []
    for name in paths:
        r = by_source[name]
        overview.append(dict(dataset=name, raw_records=stats[name]['total'],
                    valid_records=stats[name]['valid'], invalid_records=stats[name]['total']-stats[name]['valid'],
                    unique_coding_CDS=len(keys[name]), unique_proteins=len(proteins[name]),
                    duplicate_valid_rows=len(r)-len(keys[name]),
                    overlap_test_CDS=len(keys[name] & test_keys) if name != 'test' else 0,
                    overlap_test_proteins=len(proteins[name] & test_proteins) if name != 'test' else 0))
    df = pd.DataFrame(overview)
    df.to_csv(out/'01_dataset_overview.csv', index=False)
    pd.DataFrame([{'dataset':n,'reason':reason,'count':c} for n,ctr in errors.items() for reason,c in ctr.items()],
                 columns=['dataset','reason','count']).to_csv(out/'02_invalid_reasons.csv',index=False)
    pd.DataFrame([{'accession':k,'count_valid_rows':v,'organism_verified':'NO'} for k,v in acc.most_common()]).to_csv(out/'03_CAI_accessions_NEED_TAXONOMY_CHECK.csv',index=False)
    pd.DataFrame([{'dataset_a':a,'dataset_b':b,
                  'shared_CDS':len(keys[a]&keys[b]),'shared_proteins':len(proteins[a]&proteins[b])}
                  for i,a in enumerate(paths) for b in list(paths)[i+1:]]).to_csv(out/'04_cross_dataset_overlap.csv',index=False)

    # Safe default: do not use CAI as 'E. coli' without genome provenance verification.
    permitted = ['finetune'] + (['CAI'] if args.allow_unverified_cai else [])
    candidates = [r for r in records if r['source'] in permitted and
                  r['coding_dna'] not in test_keys and r['protein'] not in test_proteins]
    # Dedupe using coding sequence; retain original sequence and provenance.
    unique = {}
    for r in candidates:
        unique.setdefault(r['coding_dna'],r)
    kept = list(unique.values())
    pd.DataFrame([dict(source=r['source'], record_id=r['record_id'],
                       dna_sequence=r['dna'], protein_sequence=r['protein'],
                       n_amino_acids=len(r['protein']), terminal_stop=r['terminal_stop'],
                       accession=r['accession'], provided_cai=r['provided_cai'])
                  for r in kept],columns=['source','record_id','dna_sequence','protein_sequence',
                                          'n_amino_acids','terminal_stop','accession','provided_cai']).to_csv(
                                          out/'05_candidate_training_CDS.csv',index=False)
    pd.DataFrame([dict(source=r['source'],record_id=r['record_id'],
                       dna_sequence=r['dna'],protein_sequence=r['protein']) for r in by_source['test']],
                 columns=['source','record_id','dna_sequence','protein_sequence']).drop_duplicates(subset=['dna_sequence']).to_csv(
                 out/'06_reserved_test_CDS.csv', index=False)
    print('\n=== Dataset audit ===\n'+df.to_string(index=False))
    print('\n=== Overlap with test set ===')
    for name in ['CAI','finetune']:
        print(f'{name}: exact-CDS={len(keys[name]&test_keys)}; same protein={len(proteins[name]&test_proteins)}')
    print(f'\nEligible for conditional fine-tuning (default provenance policy): {len(kept)} distinct CDS, '
          f'{len({r["protein"] for r in kept})} distinct proteins')
    if not args.allow_unverified_cai:
        print('CAI.csv excluded from training candidates until accession->species verification.')
    print('CAUTION: Protein-level split still required between train/val/test; homologs should be clustered.')
    print('CAUTION: All counts reflect local file content only, not independent wet-lab expression samples.')
    print(f'\nSaved audit CSVs to {out}')

if __name__=='__main__': main()
