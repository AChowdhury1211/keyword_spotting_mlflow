#!/usr/bin/env python3

import warnings
import hydra
from keras.models import Sequential
from hydra.core.config_store import ConfigStore
from config_dir.configType import KWSConfig
from src import train
from src import data
from src.model import CNN_LSTM_Model
from src.experiment_tracking import MLFlowTracker, ModelSelection
warnings.filterwarnings('ignore')

cs = ConfigStore.instance()
cs.store(name="kws_config", node=KWSConfig)

@hydra.main(config_path="config_dir", config_name="config")
def main(cfg: KWSConfig) -> None:
    try:
        
        tracker = MLFlowTracker(cfg.names.experiment_name, cfg.paths.mlflow_tracking_uri)
        tracker.log()
       
        
        dataset_ = data.Dataset()
        preprocess_ = data.Preprocess(dataset_,cfg.paths.train_dir, cfg.params.n_mfcc,
                                cfg.params.mfcc_length, cfg.params.sampling_rate)                   
        preprocessed_dataset: data.Dataset = preprocess_.preprocess_dataset(preprocess_.labels,
                                                    cfg.params.test_data_split_percent)
        [data.print_shape(key, value) for key, value in preprocessed_dataset.__dict__.items()]

        
        model: Sequential = CNN_LSTM_Model((cfg.params.n_mfcc, cfg.params.mfcc_length),
                                        len(preprocess_.labels)).create_model()
        best_selected_model: ModelSelection  = train.Training(model, preprocessed_dataset,
                                                             cfg.params.batch_size,
                                                             cfg.params.epochs,
                                                             cfg.params.learning_rate,
                                                             tracker,
                                                             cfg.names.metric_name).train()

    except Exception as exc:
        raise Exception("ffhffh") from exc
                         
if __name__ == "__main__":
    main()