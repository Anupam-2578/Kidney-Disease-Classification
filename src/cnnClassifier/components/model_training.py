from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import ModelTrainingConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import pickle


class ModelTraining:
    def __init__(self , config : ModelTrainingConfig):
        self.config = config
    
    def train_val_generators(self):
        """
        Creates the training and validation data generators

        Args:
            TRAINING_DIR (string): directory path containing the training images
            VALIDATION_DIR (string): directory path containing the testing/validation images

        Returns:
            train_generator, validation_generator - tuple containing the generators
        """
        TRAINING_DIR = self.config.training_directory
        VALIDATION_DIR = self.config.validation_directory
        # Instantiate the ImageDataGenerator class (and set the arguments to agument the images)
        train_datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range=45,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')
        # Pass the appropriate arguments to the flow_from_directory method for the training data
        train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                            batch_size=16,
                                                            class_mode='categorical',
                                                            target_size=(512 , 512))

        # Instantiate the ImageDataGenerator class (with rescale)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # pass the appropriate argument to the flow_from_directory method for the trainig data
        validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                        batch_size=16,
                                                                        class_mode='categorical',
                                                                        target_size=(512, 512))

        # Return: Train, Validation
        return train_generator, validation_generator

    # Test Generators
    
    
    def train_model(self):
        self.train_generator, self.validation_generator = self.train_val_generators()
        model = load_model(self.config.untrained_model_path)
        
        logger.info(f'Model Summary :========> {model.summary()} ')
        logger.info(f'Model Training Started')
        
        history = model.fit(self.train_generator,
                             epochs=10,
                             verbose=2,
                             validation_data=self.validation_generator)
        
        model.save(self.config.trained_model_path)
        history_path = os.path.join(self.config.root_dir,'history.pkl')
        with open(history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
        logger.info(f'Model Training Completed and model saved at {self.config.trained_model_path}')
        