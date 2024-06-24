from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
import matplotlib.pyplot as plt
import os

class ModelEvaluation:
    def __init__(self , config : EvaluationConfig):
        self.config = config

    def evaluate_model(self):
        #-----------------------------------------------------------
        # Retrieve a list of list results on training and test data
        # sets for each training epoch
        #-----------------------------------------------------------
        history = self.config.model_history_path
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(len(acc)) # Get number of epochs

        #------------------------------------------------
        # Plot training and validation accuracy per epoch
        #------------------------------------------------
        plt.plot(epochs, acc, 'r', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        plt.title('Training and validation accuracy')
        plt.legend()
        accuracy_plot_path = os.path.join(self.config.root_dir, 'accuracy_plot.jpg')
        plt.savefig(accuracy_plot_path)
        plt.close() 
        

        #------------------------------------------------
        # Plot training and validation loss per epoch
        #------------------------------------------------
        plt.plot(epochs, loss, 'r', label="Training Loss")
        plt.plot(epochs, val_loss, 'b', label="Validation Loss")
        plt.title('Training and validation loss')
        plt.legend()
        loss_plot_path = os.path.join(self.config.root_dir, 'loss_plot.jpg')
        plt.savefig(loss_plot_path)
        plt.close()  
        
        logger.info("Model Evaluation Completed")
