import os
import hydra
import mlflow
import numpy as np
from typing import Any, Tuple
from config_dir.configType import KWSConfig
from src import data
from src.exception_handler import NotFoundError, DirectoryError, ValueError

class KeywordSpotter:
    def __init__(self, audio_file: str, model_artifactory_dir: str,
                n_mfcc: int, mfcc_length: int, sampling_rate: int) -> None:
        self.audio_file = audio_file
        self.model_artifactory_dir = model_artifactory_dir
        self.n_mfcc = n_mfcc
        self.mfcc_length = mfcc_length
        self.sampling_rate = sampling_rate

    def predict(self) -> Tuple[str, float]:
        
        try:
            if not os.path.exists(self.model_artifactory_dir):
                raise DirectoryError(
                       f"{self.model_artifactory_dir} doesn't exists. Please enter a valid path !!!"
                        )

            model: Any = mlflow.keras.load_model(self.model_artifactory_dir)
            audio_mfcc: np.ndarray = data.convert_audio_to_mfcc(self.audio_file,
                                                    self.n_mfcc,
                                                    self.mfcc_length,
                                                    self.sampling_rate)
            reshaped_audio_mfcc: np.ndarray = audio_mfcc.reshape(1, 49, 40)
            labels = data.Preprocess().wrap_labels()
            model_output = model.predict(reshaped_audio_mfcc)
            predicted_keyword: str = labels[np.argmax(model_output)]
            label_probability: float = max([round(value,4) for value in 
                                list(dict(enumerate(model_output.flatten(), 1)).values())])
           
            if predicted_keyword is None or label_probability is None :
                raise ValueError("Model returned empty predictions!!!")
           
            return predicted_keyword, label_probability

        except NotFoundError as exc:
            raise NotFoundError(f"Cannot infer from model. Please check the paths and try it again !!! {exc}") from exc

@hydra.main(config_path="config_dir", config_name="config")
def main(cfg: KWSConfig) -> None:
    inference_ = KeywordSpotter(cfg.names.audio_file,
                                cfg.paths.model_artifactory_dir,
                                cfg.params.n_mfcc, 
                                cfg.params.mfcc_length, cfg.params.sampling_rate)
    predicted_keyword, label_probability = inference_.predict()
    print(f"Predicted keyword: {predicted_keyword} \n Keyword probability: {label_probability}")

if __name__ == "__main__":
    main()