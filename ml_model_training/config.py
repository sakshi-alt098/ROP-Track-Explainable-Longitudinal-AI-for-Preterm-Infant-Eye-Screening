"""
Module 1: Configuration, Hyperparameters and Class Mappings for ROP Detection
"""

import os
from pathlib import Path

class ROPConfig:
    # Model target hyperparameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 18
    LR = 8e-4
    SEED = 24
    
    # 10 clinical target categories from reference study
    CATEGORIES = ['DG0', 'DG1', 'DG10', 'DG11', 'DG12', 'DG13', 'DG2', 'DG3', 'DG8', 'DG9']
    
    CLASS_NAMES = {
        'DG0':  'No ROP (Normal Retina)',
        'DG1':  'Stage 1 ROP (Demarcation Line)',
        'DG2':  'Stage 2 ROP (Intraretinal Ridge)',
        'DG3':  'Stage 3 ROP (Extraretinal Proliferation)',
        'DG8':  'Plus Disease (Vascular Tortuosity)',
        'DG9':  'Pre-Plus Disease',
        'DG10': 'Aggressive Posterior ROP',
        'DG11': 'Post-Treatment / Inactive',
        'DG12': 'Regressed ROP',
        'DG13': 'Stage 5 ROP (Total Detachment)'
    }
    
    # Local paths
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    GRADCAM_DIR = OUTPUT_DIR / "gradcam_viz"
    RESULTS_DIR = OUTPUT_DIR / "results"

cfg = ROPConfig()
for d in [cfg.OUTPUT_DIR, cfg.MODEL_DIR, cfg.GRADCAM_DIR, cfg.RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
