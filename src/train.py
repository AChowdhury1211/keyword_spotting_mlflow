

from tensorflow import keras
from keras import optimizers
from src.model import CNN_LSTM_Model
from src.data import Dataset
from src.exception_handler import ValueError
from src.experiment_tracking import MLFlowTracker, ModelSelection

class Training:
    def __init__(self, model: CNN_LSTM_Model, dataset: Dataset,
                batch_size: int, epochs: int, learning_rate: float,
                tracker: MLFlowTracker, metric_name: str) -> None:
      
        self.model = model
        self.dataset_ = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tracker = tracker
        self.metric_name = metric_name
        
    def train(self) -> ModelSelection:
        
        if self.metric_name is None:
            raise ValueError("Please provide the metric name for model selection !!!")
            
        print("Training started.....")
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Nadam(learning_rate=self.learning_rate),
                        metrics=['accuracy'])
                        
        history = self.model.fit(self.dataset_.x_train, self.dataset_.y_train,
                        batch_size = self.batch_size,
                        epochs = self.epochs,
                        verbose = 1,
                        validation_data = (self.dataset_.x_test, self.dataset_.y_test))

        return ModelSelection(self.tracker.find_best_model(self.metric_name))