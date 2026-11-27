#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare E. coli CDS/protein pairs and four RNA features for CodonTransformer.

Sources:
  1) CodonTransformer/dataset.csv (large multi-species CSV; streamed in chunks)
  2) ColiFormer-Data/audit_results/05_candidate_training_CDS.csv
  3) ColiFormer-Data/CAI.csv (accession taxonomy is unverified in the supplied audit;
     quarantined by default unless --include-unverified-cai is specified)
  4) ColiFormer-Data/audit_results/06_reserved_test_CDS.csv (reserved test only)

Features, matching the user's prior E. coli analysis conventions:
  CAI: Biopython CodonAdaptationIndex built from ecol.heg.fasta
  Global_MFE: ViennaRNA RNAfold MFE of the full CDS (CDS-only context)
  MFE30: ViennaRNA RNAfold MFE of CDS nt 1..30
  OE20: -RT ln(P(unpaired for CDS nt 1..20)), RNAplfold -W 150 -L 100 -u 30 -T 37

The script does NOT deduplicate by protein. Distinct synonymous CDS for the same
protein are retained. All CDS encoding a reserved test protein are excluded from
training to prevent protein-level leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex

# ----------------------------- Defaults ------------------------------------
ROOT = Path('/mnt/sdc/guansheng/protein_RNA/clear_understand')
DATA = ROOT / 'dataset'
CT_CSV = DATA / 'CodonTransformer' / 'dataset.csv'
COLI = DATA / 'ColiFormer-Data'
CANDIDATE_CSV = COLI / 'audit_results' / '05_candidate_training_CDS.csv'
CAI_CSV = COLI / 'CAI.csv'
TAXONOMY_CSV = COLI / 'audit_results' / '03_CAI_accessions_NEED_TAXONOMY_CHECK.csv'
TEST_CSV = COLI / 'audit_results' / '06_reserved_test_CDS.csv'
HEG_FASTA = Path('/mnt/sdc/guansheng/protein_RNA/dataset/ecol.heg.fasta/ecol.heg.fasta')
OUTDIR = ROOT / 'data_preparation' / 'ecoli_codontransformer'

R_KCAL = 0.00198720425864083


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ct-csv', type=Path, default=CT_CSV)
    p.add_argument('--coli-dir', type=Path, default=COLI)
    p.add_argument('--candidate-csv', type=Path, default=CANDIDATE_CSV)
    p.add_argument('--cai-csv', type=Path, default=CAI_CSV)
    p.add_argument('--taxonomy-csv', type=Path, default=TAXONOMY_CSV)
    p.add_argument('--test-csv', type=Path, default=TEST_CSV)
    p.add_argument('--heg-fasta', type=Path, default=HEG_FASTA)
    p.add_argument('--outdir', type=Path, default=OUTDIR)
    p.add_argument('--chunksize', type=int, default=50000)
    p.add_argument('--temperature', type=float, default=37.0)
    p.add_argument('--rnaplfold-window', type=int, default=150)
    p.add_argument('--rnaplfold-span', type=int, default=100)
    p.add_argument('--organism-id', type=int, default=51,
                   help='CodonTransformer E. coli organism ID; 51 is the ID in the supplied ColiFormer JSON.')
    p.add_argument('--organism-map', type=Path,
                   default=Path('/home/guansheng/code_protein_RNA/code/CodonTransformer-main/src/organism2id.pkl'),
                   help='Official CodonTransformer organism2id.pkl; used to check the E. coli ID when available.')
    p.add_argument('--include-unverified-cai', action='store_true',
                   help='Include CAI.csv accessions marked NO/unverified in the training-eligible pool. Not recommended until taxonomy is checked.')
    p.add_argument('--max-official-rows', type=int, default=0,
                   help='Debug only: inspect at most this many official CSV rows; 0 means all rows.')
    p.add_argument('--force-features', action='store_true',
                   help='Recompute all RNA features instead of using SQLite cache.')
    return p.parse_args()


def clean_dna(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = re.sub(r'\s+', '', str(value)).upper().replace('U', 'T')
    if not s or re.fullmatch(r'[ACGT]+', s) is None:
        return None
    if len(s) < 30:
        return None
    if len(s) % 3 != 0:
        return None
    return s


def translate_complete_cds(dna: str) -> str:
    """Translate a complete bacterial CDS using NCBI bacterial table 11.

    cds=True verifies a valid start, one terminal stop, and no internal stops.
    It also applies bacterial alternative-start semantics (initial aa = M).
    """
    return str(Seq(dna).translate(table=11, cds=True))


def protein_clean(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = re.sub(r'\s+', '', str(value)).upper().rstrip('*_')
    if not s or re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', s) is None:
        return None
    return s


def make_tokens(dna: str, protein: str) -> str:
    codons = [dna[i:i+3] for i in range(0, len(dna), 3)]
    if len(codons) != len(protein) + 1:
        raise ValueError('Expected one terminal stop codon after the amino-acid codons')
    aa_tokens = [f'{aa}_{codon}' for aa, codon in zip(protein, codons[:-1])]
    stop = codons[-1]
    if stop not in {'TAA', 'TAG', 'TGA'}:
        raise ValueError(f'Invalid terminal stop: {stop}')
    return ' '.join(aa_tokens + [f'__{stop}'])


def load_taxonomy_map(path: Path) -> Dict[str, str]:
    """Return accession -> YES/NO from the user's accession audit file."""
    result = {}
    if not path.exists():
        return result
    df = pd.read_csv(path, dtype=str)
    if not {'accession', 'organism_verified'}.issubset(df.columns):
        return result
    for row in df.itertuples(index=False):
        accession = str(getattr(row, 'accession')).strip()
        status = str(getattr(row, 'organism_verified')).strip().upper()
        if accession and accession.lower() != 'nan':
            result[accession] = status
    return result


def add_record(store: Dict[str, dict], record: dict):
    dna = record['dna_sequence']
    if dna not in store:
        record['source_datasets'] = {record['source_dataset']}
        store[dna] = record
        return

    old = store[dna]
    if old['protein_sequence'] != record['protein_sequence']:
        raise ValueError('Same CDS translated to different proteins; investigate genetic code/data corruption')
    old['source_datasets'].add(record['source_dataset'])
    # Preserve all accession IDs and source organism labels for provenance.
    for key in ('accession', 'source_organism', 'source_strain'):
        a = str(old.get(key, '') or '').strip()
        b = str(record.get(key, '') or '').strip()
        if b and b.lower() != 'nan' and b not in a.split('; '):
            old[key] = (a + '; ' + b).strip('; ')
    # Keep a supplied CAI for audit only; it is never used as the unified target.
    if old.get('provided_cai') is None and record.get('provided_cai') is not None:
        old['provided_cai'] = record['provided_cai']
    if old.get('taxonomy_status') == 'unverified_accession' and record.get('taxonomy_status') != 'unverified_accession':
        old['taxonomy_status'] = record.get('taxonomy_status')


def read_reserved_test(path: Path) -> Tuple[dict, set, set]:
    if not path.exists():
        raise FileNotFoundError(f'Reserved test CSV not found: {path}')
    df = pd.read_csv(path, dtype=str, low_memory=False)
    required = {'dna_sequence', 'protein_sequence'}
    if not required.issubset(df.columns):
        raise ValueError(f'Test CSV missing columns: {required - set(df.columns)}')
    records = {}
    test_dna, test_proteins = set(), set()
    for i, row in df.iterrows():
        dna = clean_dna(row['dna_sequence'])
        if dna is None:
            raise ValueError(f'Invalid reserved test CDS at row {i}')
        translated = translate_complete_cds(dna)
        supplied = protein_clean(row['protein_sequence'])
        if supplied and supplied != translated:
            raise ValueError(f'Reserved test protein/CDS mismatch at row {i}')
        if len(translated) + 1 > 2046:
            raise ValueError(f'Reserved test record exceeds model token limit at row {i}')
        rec = {
            'record_id': str(row.get('record_id', i)),
            'dna_sequence': dna,
            'protein_sequence': translated,
            'source_dataset': 'ColiFormer_reserved_test',
            'source_organism': 'Escherichia coli (ColiFormer organism ID 51)',
            'source_strain': '',
            'taxonomy_status': 'dataset_label_organism_51',
            'provided_cai': None,
            'accession': '',
        }
        records[dna] = rec
        test_dna.add(dna)
        test_proteins.add(translated)
    return records, test_dna, test_proteins


def read_official_ecoli(path: Path, chunksize: int, max_rows: int = 0) -> Dict[str, dict]:
    """Stream 4.4-GB CodonTransformer CSV and retain E. coli rows only."""
    if not path.exists():
        raise FileNotFoundError(path)
    print(f'[OFFICIAL] Streaming {path} with chunksize={chunksize}', flush=True)
    store = {}
    rows_seen = ecoli_rows = invalid = mismatch = too_long = 0
    mismatch_samples = []
    usecols = ['dna', 'protein', 'organism', 'GeneID', 'description']
    reader = pd.read_csv(path, usecols=lambda c: c in usecols,
                         dtype=str, chunksize=chunksize, low_memory=False)
    for chunk_no, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        org = chunk['organism'].fillna('').astype(str)
        mask = org.str.contains(r'\bEscherichia\s+coli\b', case=False, regex=True)
        sub = chunk.loc[mask]
        ecoli_rows += len(sub)
        for row in sub.itertuples(index=False):
            values = row._asdict()
            dna = clean_dna(values.get('dna'))
            if dna is None:
                invalid += 1
                continue
            try:
                translated = translate_complete_cds(dna)
            except Exception:
                invalid += 1
                continue
            supplied = protein_clean(values.get('protein'))
            if supplied is None or supplied != translated:
                mismatch += 1
                if len(mismatch_samples) < 8:
                    raw = str(values.get('protein', ''))
                    mismatch_samples.append({'dna_len': len(dna), 'provided_len': len(raw), 'provided_head': raw[:35], 'provided_tail': raw[-40:], 'translated_len': len(translated), 'translated_head': translated[:35], 'translated_tail': translated[-40:]})
                continue
            if len(translated) + 1 > 2046:
                too_long += 1
                continue
            org_name = str(values.get('organism') or '')
            rec = {
                'record_id': str(values.get('GeneID') or ''),
                'dna_sequence': dna,
                'protein_sequence': translated,
                'source_dataset': 'CodonTransformer_official_Ecoli',
                'source_organism': org_name,
                'source_strain': org_name,
                'taxonomy_status': 'official_organism_field_E_coli',
                'provided_cai': None,
                'accession': str(values.get('description') or ''),
            }
            add_record(store, rec)
        if chunk_no % 5 == 0:
            print(f'[OFFICIAL] chunks={chunk_no}, rows_seen={rows_seen:,}, Ecoli_rows={ecoli_rows:,}, unique_CDS={len(store):,}', flush=True)
        if max_rows and rows_seen >= max_rows:
            print(f'[OFFICIAL] DEBUG STOP at {rows_seen} rows', flush=True)
            break
    print(f'[OFFICIAL] rows_seen={rows_seen:,}; ecoli_rows={ecoli_rows:,}; unique_CDS={len(store):,}; invalid={invalid:,}; protein_mismatch={mismatch:,}; over_length={too_long:,}', flush=True)
    if mismatch_samples:
        print('[OFFICIAL] First mismatch examples (truncated):', flush=True)
        for item in mismatch_samples:
            print(json.dumps(item, ensure_ascii=False), flush=True)
    return store


def read_coliformer_candidate(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    required = {'dna_sequence', 'protein_sequence'}
    if not required.issubset(df.columns):
        raise ValueError(f'Candidate CSV missing columns: {required - set(df.columns)}')
    store = {}
    bad = mismatch = long = 0
    for i, row in df.iterrows():
        dna = clean_dna(row['dna_sequence'])
        if dna is None:
            bad += 1
            continue
        try:
            translated = translate_complete_cds(dna)
        except Exception:
            bad += 1
            continue
        supplied = protein_clean(row['protein_sequence'])
        if supplied is None or supplied != translated:
            mismatch += 1
            continue
        if len(translated) + 1 > 2046:
            long += 1
            continue
        rec = {
            'record_id': str(row.get('record_id', i)),
            'dna_sequence': dna,
            'protein_sequence': translated,
            'source_dataset': 'ColiFormer_finetune',
            'source_organism': 'Escherichia coli (ColiFormer organism ID 51)',
            'source_strain': '',
            'taxonomy_status': 'dataset_label_organism_51',
            'provided_cai': pd.to_numeric(row.get('provided_cai'), errors='coerce'),
            'accession': str(row.get('accession') or ''),
        }
        add_record(store, rec)
    print(f'[COLIFORMER finetune] unique_CDS={len(store):,}; invalid={bad}; protein_mismatch={mismatch}; over_length={long}', flush=True)
    return store


def read_coliformer_cai(path: Path, taxonomy_path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Return (verified_accession records, unverified/quarantine records)."""
    if not path.exists():
        raise FileNotFoundError(path)
    tax_map = load_taxonomy_map(taxonomy_path)
    verified, unverified = {}, {}
    bad = mismatch = long = 0
    # CAI.csv is only ~45 MB; normal read is safe.
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if 'Sequence' not in df.columns:
        raise ValueError('CAI.csv has no Sequence column')
    id_col = 'Unnamed: 0' if 'Unnamed: 0' in df.columns else df.columns[0]
    for i, row in df.iterrows():
        dna = clean_dna(row.get('Sequence'))
        if dna is None:
            bad += 1
            continue
        try:
            translated = translate_complete_cds(dna)
        except Exception:
            bad += 1
            continue
        if len(translated) + 1 > 2046:
            long += 1
            continue
        record_id = str(row.get(id_col, i))
        accession = record_id.split('_cds_')[0] if '_cds_' in record_id else ''
        status = tax_map.get(accession, 'NO')
        provided_cai = pd.to_numeric(row.get('CAI'), errors='coerce')
        rec = {
            'record_id': record_id,
            'dna_sequence': dna,
            'protein_sequence': translated,
            'source_dataset': 'ColiFormer_CAI',
            'source_organism': 'ColiFormer CAI dataset; accession taxonomy pending',
            'source_strain': '',
            'taxonomy_status': 'verified' if status == 'YES' else 'unverified_accession',
            'provided_cai': float(provided_cai) if pd.notna(provided_cai) else None,
            'accession': accession,
        }
        target = verified if status == 'YES' else unverified
        add_record(target, rec)
    print(f'[COLIFORMER CAI] verified_unique_CDS={len(verified):,}; unverified_unique_CDS={len(unverified):,}; invalid={bad}; over_length={long}', flush=True)
    return verified, unverified


def build_cai_reference(fasta: Path):
    if not fasta.exists():
        raise FileNotFoundError(f'CAI reference FASTA not found: {fasta}')
    seqs = []
    for rec in SeqIO.parse(str(fasta), 'fasta'):
        s = str(rec.seq).upper().replace('U', 'T')
        if s and len(s) % 3 == 0 and re.fullmatch(r'[ACGT]+', s):
            # The HEG reference may contain CDS without a stop. CAI calculation
            # uses the codon string; invalid/internal-stop records are skipped.
            try:
                aa = str(Seq(s).translate(table=11, to_stop=False))
                if '*' in aa.rstrip('*'):
                    continue
                seqs.append(s)
            except Exception:
                continue
    if len(seqs) < 10:
        raise ValueError(f'Too few usable CDS in CAI reference: {len(seqs)}')
    cai = CodonAdaptationIndex(seqs)
    print(f'[CAI] Loaded {len(seqs):,} reference CDS from {fasta}', flush=True)
    return cai, len(seqs)


def open_feature_cache(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute('''CREATE TABLE IF NOT EXISTS features (
        dna TEXT PRIMARY KEY,
        cai REAL,
        global_mfe REAL,
        mfe30 REAL,
        popen20 REAL,
        oe20 REAL,
        temperature REAL,
        window INTEGER,
        span INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    return conn


def call_rnaplfold(dna: str, temperature: float, window: int, span: int) -> Tuple[float, float]:
    rna = dna.replace('T', 'U')
    cmd = ['RNAplfold', '-W', str(window), '-L', str(span), '-u', '30', '-T', str(temperature)]
    with tempfile.TemporaryDirectory(prefix='ecoli_rnaplfold_') as tmp:
        proc = subprocess.run(cmd, input='>candidate\n' + rna + '\n', text=True,
                              capture_output=True, cwd=tmp, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f'RNAplfold failed ({proc.returncode}): {proc.stderr[-1000:]}')
        files = list(Path(tmp).glob('*_lunp'))
        if len(files) != 1:
            raise RuntimeError(f'Expected one *_lunp file, found {[x.name for x in files]}')
        row20 = None
        with open(files[0], 'r', encoding='utf-8') as fh:
            for line in fh:
                cells = line.split()
                if cells and cells[0].isdigit() and int(cells[0]) == 20:
                    row20 = cells
                    break
        if row20 is None or len(row20) <= 20:
            raise RuntimeError('RNAplfold _lunp lacks P(unpaired for first 20 nt) at row 20')
        popen20 = float(row20[20])
        if not np.isfinite(popen20) or not (0 < popen20 <= 1):
            raise ValueError(f'Invalid Popen20={popen20}')
        oe20 = -R_KCAL * (temperature + 273.15) * math.log(popen20)
        return popen20, float(oe20)


def compute_one(dna: str, cai_model, temperature: float, window: int, span: int) -> dict:
    import RNA
    RNA.cvar.temperature = float(temperature)
    rna = dna.replace('T', 'U')
    fc = RNA.fold_compound(rna)
    global_mfe = float(fc.mfe()[1])
    mfe30 = float(RNA.fold(rna[:30])[1])
    cai_value = float(cai_model.calculate(dna))
    popen20, oe20 = call_rnaplfold(dna, temperature, window, span)
    vals = [cai_value, global_mfe, mfe30, popen20, oe20]
    if not all(np.isfinite(x) for x in vals):
        raise ValueError('Non-finite feature value')
    return {
        'CAI': cai_value,
        'Global_MFE': global_mfe,
        'MFE30': mfe30,
        'Popen20': popen20,
        'OE20': oe20,
    }


def get_or_compute(dna: str, cai_model, conn, args) -> dict:
    if not args.force_features:
        row = conn.execute(
            'SELECT cai, global_mfe, mfe30, popen20, oe20 FROM features WHERE dna=? AND temperature=? AND window=? AND span=?',
            (dna, args.temperature, args.rnaplfold_window, args.rnaplfold_span)
        ).fetchone()
        if row and all(v is not None and np.isfinite(v) for v in row):
            return dict(zip(['CAI', 'Global_MFE', 'MFE30', 'Popen20', 'OE20'], map(float, row)))
    feat = compute_one(dna, cai_model, args.temperature, args.rnaplfold_window, args.rnaplfold_span)
    conn.execute('''INSERT OR REPLACE INTO features
        (dna,cai,global_mfe,mfe30,popen20,oe20,temperature,window,span,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)''',
        (dna, feat['CAI'], feat['Global_MFE'], feat['MFE30'], feat['Popen20'], feat['OE20'],
         args.temperature, args.rnaplfold_window, args.rnaplfold_span))
    return feat


def stable_split(protein: str) -> str:
    # Deterministic protein-level split: all synonymous CDS of one protein stay together.
    h = int(hashlib.sha1(protein.encode('ascii')).hexdigest()[:8], 16)
    return 'val' if h % 10 == 0 else 'train'


def write_jsonl(path: Path, df: pd.DataFrame, organism_id: int, extended: bool,
                scaler: Optional[dict] = None):
    with open(path, 'w', encoding='utf-8') as out:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            d = row._asdict()
            rec = {'idx': int(i), 'codons': d['tokenized'], 'organism': int(organism_id)}
            if extended:
                targets = {k: float(d[k]) for k in ['CAI', 'Global_MFE', 'MFE30', 'OE20']}
                rec.update({
                    'protein_sequence': d['protein_sequence'],
                    'dna_sequence': d['dna_sequence'],
                    'targets': targets,
                    'source_dataset': d['source_datasets'],
                    'protein_group': d['protein_group'],
                })
                if scaler:
                    rec['targets_z'] = {
                        k: (targets[k] - scaler[k]['mean']) / scaler[k]['std']
                        for k in targets
                    }
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')



def verify_organism_id(mapping_path: Path, organism_id: int):
    """Best-effort validation of the E. coli token_type_id against official pickle."""
    if not mapping_path.exists():
        print(f'[ORGANISM MAP] WARNING: mapping file not found: {mapping_path}; using supplied ColiFormer organism ID={organism_id}', flush=True)
        return
    try:
        with open(mapping_path, 'rb') as fh:
            mapping = pickle.load(fh)
    except Exception as exc:
        print(f'[ORGANISM MAP] WARNING: cannot read mapping ({exc}); using ID={organism_id}', flush=True)
        return
    found = set()
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if isinstance(key, str) and 'escherichia coli' in key.lower():
                try:
                    found.add(int(value))
                except Exception:
                    pass
            if isinstance(value, str) and 'escherichia coli' in value.lower():
                try:
                    found.add(int(key))
                except Exception:
                    pass
    if found:
        print(f'[ORGANISM MAP] E. coli IDs found in official map: {sorted(found)}', flush=True)
        if organism_id not in found:
            raise ValueError(f'--organism-id={organism_id} does not match E. coli ID(s) in organism2id.pkl: {sorted(found)}. Please use the official mapping ID.')
    else:
        print(f'[ORGANISM MAP] WARNING: no E. coli string found in mapping structure; using supplied ID={organism_id}. Check pickle manually.', flush=True)


def write_model_csv(path: Path, df: pd.DataFrame, organism_id: int):
    """Write a CSV mirroring the visible official dataset.csv column layout."""
    out = pd.DataFrame({
        'dna': df['dna_sequence'],
        'protein': df['protein_sequence'],
        'organism': 'Escherichia coli',
        'GeneID': df['record_id'].astype(str),
        'description': df['source_datasets'].astype(str),
        'tokenized': df['tokenized'],
        'fine_tuned': True,
        # The official dataset's len is codon-token count (CDS nt length / 3).
        'len': (df['dna_sequence'].str.len() // 3).astype(int),
        'organism_id': int(organism_id),
        'CAI': df['CAI'],
        'Global_MFE': df['Global_MFE'],
        'MFE30': df['MFE30'],
        'OE20': df['OE20'],
        'protein_group': df['protein_group'],
        'split': df['split'],
    })
    out.to_csv(path, index=False)

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    verify_organism_id(args.organism_map, args.organism_id)
    for exe in ('RNAfold', 'RNAplfold'):
        if not shutil.which(exe):
            raise FileNotFoundError(f'{exe} not found in PATH. Activate the protein_RNA environment with ViennaRNA installed.')
    if args.rnaplfold_window < args.rnaplfold_span or args.rnaplfold_span < 30:
        raise ValueError('Require RNAplfold window >= span >= 30')

    print('[1/7] Read reserved test set', flush=True)
    test_records, test_dna, test_proteins = read_reserved_test(args.test_csv)
    print(f'[TEST] reserved CDS={len(test_dna):,}; reserved proteins={len(test_proteins):,}', flush=True)

    print('[2/7] Read official CodonTransformer E. coli subset', flush=True)
    official = read_official_ecoli(args.ct_csv, args.chunksize, args.max_official_rows)

    print('[3/7] Read ColiFormer fine-tuning candidates', flush=True)
    finetune = read_coliformer_candidate(args.candidate_csv)

    print('[4/7] Read ColiFormer CAI candidates and accession audit', flush=True)
    cai_verified, cai_unverified = read_coliformer_cai(args.cai_csv, args.taxonomy_csv)

    # Merge by exact CDS. Keep distinct synonymous CDS for the same protein.
    merged: Dict[str, dict] = {}
    for source in (official, finetune, cai_verified, cai_unverified):
        for rec in source.values():
            add_record(merged, dict(rec))

    # Reserve all 100 test proteins: exclude any train candidate encoding one of them.
    excluded_test_cds = 0
    excluded_test_proteins = 0
    candidates = []
    for dna, rec in merged.items():
        if dna in test_dna:
            excluded_test_cds += 1
            continue
        if rec['protein_sequence'] in test_proteins:
            excluded_test_proteins += 1
            continue
        rec['source_datasets'] = ';'.join(sorted(rec['source_datasets']))
        rec['protein_group'] = hashlib.sha1(rec['protein_sequence'].encode('ascii')).hexdigest()
        rec['target_host'] = 'Escherichia coli general'
        rec['organism_id'] = int(args.organism_id)
        rec['CDS_RNA'] = dna.replace('T', 'U')
        rec['CDS_nt_length'] = len(dna)
        rec['protein_aa_length'] = len(rec['protein_sequence'])
        rec['tokenized'] = make_tokens(dna, rec['protein_sequence'])
        rec['split'] = stable_split(rec['protein_sequence'])
        rec['training_eligible'] = (
            rec['taxonomy_status'] != 'unverified_accession' or args.include_unverified_cai
        )
        candidates.append(rec)
    df = pd.DataFrame(candidates)
    if df.empty:
        raise RuntimeError('No candidate CDS remain after filtering. Check E. coli organism labels and test exclusions.')
    print(f'[MERGE] unique_CDS={len(df):,}; excluded_exact_test_CDS={excluded_test_cds}; excluded_test_proteins={excluded_test_proteins}', flush=True)
    print('[MERGE] source rows by source_dataset:')
    print(df['source_datasets'].value_counts().head(20).to_string())
    print('[MERGE] taxonomy_status:')
    print(df['taxonomy_status'].value_counts(dropna=False).to_string())

    # Save pre-feature audit tables.
    df.to_csv(args.outdir / '01_ecoli_merged_candidates_pre_features.csv', index=False)
    pd.DataFrame(list(test_records.values())).to_csv(args.outdir / '00_reserved_test_records.csv', index=False)

    print('[5/7] Compute four objectives with persistent SQLite cache', flush=True)
    cai_model, n_cai_ref = build_cai_reference(args.heg_fasta)
    cache_path = args.outdir / 'feature_cache.sqlite'
    conn = open_feature_cache(cache_path)
    feature_rows = []
    failures = []
    start = time.time()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        dna = row.dna_sequence
        try:
            feat = get_or_compute(dna, cai_model, conn, args)
            feature_rows.append(feat)
        except Exception as exc:
            failures.append({'dna_sequence': dna, 'source_datasets': row.source_datasets,
                             'reason': repr(exc)})
            feature_rows.append({'CAI': np.nan, 'Global_MFE': np.nan,
                                 'MFE30': np.nan, 'Popen20': np.nan, 'OE20': np.nan})
        if i % 50 == 0 or i == len(df):
            conn.commit()
            elapsed = time.time() - start
            print(f'[FEATURES] {i:,}/{len(df):,}; failures={len(failures):,}; elapsed={elapsed/60:.1f} min', flush=True)
    # Compute the same four features for the reserved 100-record test set,
    # but keep it in a separate file and never merge it into train/validation.
    test_feature_rows = []
    for j, test_rec in enumerate(test_records.values(), start=1):
        try:
            tf = get_or_compute(test_rec['dna_sequence'], cai_model, conn, args)
            test_feature_rows.append({**test_rec, **tf, 'feature_valid': True})
        except Exception as exc:
            test_feature_rows.append({**test_rec, 'CAI': np.nan, 'Global_MFE': np.nan,
                                      'MFE30': np.nan, 'Popen20': np.nan, 'OE20': np.nan,
                                      'feature_valid': False, 'feature_error': repr(exc)})
        if j % 25 == 0 or j == len(test_records):
            conn.commit()
            print(f'[TEST FEATURES] {j:,}/{len(test_records):,}', flush=True)
    pd.DataFrame(test_feature_rows).to_csv(
        args.outdir / '14_reserved_test_four_objectives.csv', index=False
    )

    conn.commit()
    conn.close()
    feats = pd.DataFrame(feature_rows)
    result = pd.concat([df.reset_index(drop=True), feats], axis=1)
    result['feature_valid'] = np.isfinite(result[['CAI','Global_MFE','MFE30','OE20']]).all(axis=1)
    result.to_csv(args.outdir / '02_ecoli_all_candidates_four_objectives.csv', index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.outdir / '03_feature_failures.csv', index=False)

    # Primary training pool: valid features and taxonomically eligible sources.
    eligible = result.loc[result['feature_valid'] & result['training_eligible']].copy()
    eligible = eligible.sort_values(['protein_group', 'dna_sequence']).reset_index(drop=True)
    eligible.to_csv(args.outdir / '04_ecoli_training_eligible.csv', index=False)

    # Also expose the unverified CAI records separately, even though they are not in the strict train set.
    unverified = result.loc[result['taxonomy_status'].eq('unverified_accession')].copy()
    unverified.to_csv(args.outdir / '05_unverified_CAI_quarantine.csv', index=False)

    # Protein-level train/validation split is already assigned before features.
    train = eligible.loc[eligible['split'].eq('train')].copy().reset_index(drop=True)
    val = eligible.loc[eligible['split'].eq('val')].copy().reset_index(drop=True)
    if len(train) == 0:
        raise RuntimeError('Training split is empty')

    target_cols = ['CAI', 'Global_MFE', 'MFE30', 'OE20']
    scaler = {}
    for col in target_cols:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        scaler[col] = {'mean': mean, 'std': std}
    with open(args.outdir / '06_target_scaler_train_only.json', 'w', encoding='utf-8') as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    print('[6/7] Export CodonTransformer-compatible JSONL', flush=True)
    # Exact base JSONL schema seen in ColiFormer finetune_set.json.
    write_jsonl(args.outdir / '07_codontransformer_train.jsonl', train, args.organism_id, extended=False)
    write_jsonl(args.outdir / '08_codontransformer_val.jsonl', val, args.organism_id, extended=False)
    # Extended files for the planned four-objective conditional model.
    write_jsonl(args.outdir / '09_four_objective_conditional_train.jsonl', train, args.organism_id, extended=True, scaler=scaler)
    write_jsonl(args.outdir / '10_four_objective_conditional_val.jsonl', val, args.organism_id, extended=True, scaler=scaler)
    # CSV mirrors the visible CodonTransformer dataset.csv field convention and
    # additionally carries the four objective labels for the custom model.
    write_model_csv(args.outdir / '15_codontransformer_train_with_targets.csv', train, args.organism_id)
    write_model_csv(args.outdir / '16_codontransformer_val_with_targets.csv', val, args.organism_id)

    # Optional pool that includes CAI.csv accession rows still marked unverified.
    if args.include_unverified_cai:
        all_valid = result.loc[result['feature_valid']].copy()
        all_valid = all_valid.sort_values(['protein_group', 'dna_sequence']).reset_index(drop=True)
        all_valid.to_csv(args.outdir / '11_training_pool_including_unverified_CAI.csv', index=False)

    print('[7/7] Write manifest and summary', flush=True)
    summary = {
        'official_ecoli_unique_cds_before_merge': len(official),
        'coliformer_finetune_unique_cds_before_merge': len(finetune),
        'coliformer_cai_accession_verified_unique_cds': len(cai_verified),
        'coliformer_cai_accession_unverified_unique_cds': len(cai_unverified),
        'merged_unique_cds_before_test_exclusion': len(merged),
        'reserved_test_unique_cds': len(test_dna),
        'reserved_test_unique_proteins': len(test_proteins),
        'excluded_exact_test_cds': excluded_test_cds,
        'excluded_test_protein_matches': excluded_test_proteins,
        'candidate_rows_after_test_exclusion': len(df),
        'feature_valid_rows': int(result['feature_valid'].sum()),
        'feature_failures': len(failures),
        'strict_training_eligible_rows': len(eligible),
        'train_rows': len(train),
        'val_rows': len(val),
        'train_unique_proteins': int(train['protein_sequence'].nunique()),
        'val_unique_proteins': int(val['protein_sequence'].nunique()),
        'organism_id_used': int(args.organism_id),
        'target_host': 'Escherichia coli general',
        'cai_reference_fasta': str(args.heg_fasta),
        'cai_reference_cds_count': n_cai_ref,
        'temperature_C': args.temperature,
        'rnaplfold_window': args.rnaplfold_window,
        'rnaplfold_span': args.rnaplfold_span,
        'OE20_definition': '-R*T*ln(P(unpaired for CDS nt 1..20)); CDS-only context',
        'deduplication_key': 'exact DNA CDS; distinct synonymous CDS for same protein retained',
        'protein_level_split': 'SHA1(protein) first 8 hex modulo 10 == 0 => val; otherwise train',
        'unverified_CAI_included_in_strict_train': bool(args.include_unverified_cai),
        'max_codon_tokens_excluding_CLS_SEP': 2046,
    }
    with open(args.outdir / '12_data_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_csv(args.outdir / '13_data_summary.csv', index=False)

    print('\n========== FINAL SUMMARY ==========')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'\nOutputs: {args.outdir}')
    print('IMPORTANT: inspect 03_feature_failures.csv (if present), taxonomy status, and 12_data_manifest.json before training.')


if __name__ == '__main__':
    main()
