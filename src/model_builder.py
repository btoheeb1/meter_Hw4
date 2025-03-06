from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing 

#Importing the parent: ANN class from ANN.py
from src.ANN import ANN

class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)
    
    def ann_model(self, X_train, X_test, y_train, y_test):
        # Create ANN model
        ann = ANN(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=300)

        # Train the model
        ann.train(X_train, y_train)

        # Evaluate the model
        self.accuracy = ann.evaluate(X_test, y_test)

        # Plot loss curve
        ann.plot_loss_curve()

        return ann
