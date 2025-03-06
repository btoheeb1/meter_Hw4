import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class ANN:
    """
    Parent class for Artificial Neural Networks (ANNs).
    This class handles model creation, training, evaluation, and visualization.
    """
    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                   learning_rate_init=learning_rate_init, 
                                   max_iter=max_iter, 
                                   random_state=42)
    
    def train(self, X_train, y_train):
        """Trains the ANN model."""
        self.model.fit(X_train, y_train)
        print("Model Training Complete")
    
    def evaluate(self, X_test, y_test):
        """Evaluates the model on the test set."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return accuracy
    
    def plot_loss_curve(self):
        """Plots the loss curve during training."""
        plt.plot(self.model.loss_curve_)
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

