""" 
Unit test using pytest    
"""
import warnings
import pytest
import numpy as np
from omegaconf import OmegaConf
from src import data
warnings.filterwarnings('ignore')

cfg = OmegaConf.load('./config_dir/config.yaml')

@pytest.fixture
def mfcc() -> np.ndarray:
    mfcc_features = data.convert_audio_to_mfcc(cfg.names.audio_file,
                                               cfg.params.n_mfcc,
                                               cfg.params.mfcc_length,
                                               cfg.params.sampling_rate)
    return mfcc_features

def test_label_type() -> None:
    labels = data.Preprocess().wrap_labels()
    assert all(isinstance(n, str) for n in labels)

def test_mfcc_shape(mfcc: pytest.fixture) -> None:
    assert mfcc.shape == (cfg.params.n_mfcc, cfg.params.mfcc_length)

def test_mfcc_dimension(mfcc: pytest.fixture) -> None:
    assert len(mfcc.shape) == 2