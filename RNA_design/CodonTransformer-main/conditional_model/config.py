from pathlib import Path

FEATURES = ('CAI', 'Global_MFE', 'MFE30', 'OE20')
ROOT = Path('/mnt/sdc/guansheng/protein_RNA/clear_understand/data_preparation/ecoli_codontransformer')
DEFAULT_MODEL = Path('/mnt/sdc/guansheng/protein_RNA/methods/CodonTransformer-main/model')
DEFAULT_TRAIN = ROOT / '09_four_objective_conditional_train.jsonl'
DEFAULT_VAL = ROOT / '10_four_objective_conditional_val.jsonl'
DEFAULT_SCALER = ROOT / '06_target_scaler_train_only.json'
DEFAULT_OUT = Path('/mnt/sdc/guansheng/protein_RNA/clear_understand/model_training/conditional_codontransformer')
STOP = {'TAA', 'TAG', 'TGA'}
MAX_LEN = 2048
