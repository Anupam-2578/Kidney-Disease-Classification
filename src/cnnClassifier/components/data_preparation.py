import os
import random
from cnnClassifier.entity.config_entity import DatapreparationConfig
from shutil import copyfile
from cnnClassifier import logger



class DataPreparation:
    def __init__(self , config : DatapreparationConfig):
        self.config = config
        
    
        
    def create_train_val_dirs(self) -> None:
        """
        Create direcotries for the train and test sets

        Args:
        root_path (string) - the base directory path to create subdirectories from

        Returns:
        None
        """
        # Create directories for training and validation sets
        # Define the path for the train and validation sets

        self.train_path = os.path.join(self.config.root_dir, "training")
        self.val_path = os.path.join(self.config.root_dir, "validation")

        # Create the traina and validation directories
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

        # Inside each of the traina and validation directorues, create 'Cyst', 'Normal', 'Stone', 'Tumor' subdirectories
        os.makedirs(os.path.join(self.train_path, "Cyst"), exist_ok=True)
        os.makedirs(os.path.join(self.train_path, "Normal"), exist_ok=True)
        os.makedirs(os.path.join(self.train_path, "Stone"), exist_ok=True)
        os.makedirs(os.path.join(self.train_path, "Tumor"), exist_ok=True)
        os.makedirs(os.path.join(self.val_path, "Cyst"), exist_ok=True)
        os.makedirs(os.path.join(self.val_path, "Normal"), exist_ok=True)
        os.makedirs(os.path.join(self.val_path, "Stone"), exist_ok=True)
        os.makedirs(os.path.join(self.val_path, "Tumor"), exist_ok=True)
        """
        /training_validation_CDK
        |-- train
        |   |-- Cyst
        |   |-- Normal
        |   |-- Stone
        |   |-- Tumor

        |-- validation
        |   |-- Cyst
        |   |-- Normal
        |   |-- Stone
        |   |-- Tumor
        """
        logger.info(f"Directories created: {self.train_path} , {self.val_path}")
        
    # Function: split_data
    def split_data(self):
        """
        Splits the data into train and test sets

        Args:
            SOURCE_DIR (string): directory path containing the images
            TRAINING_DIR (string): directory path to be used for training
            VALIDATION_DIR (string): directory path to be used for validation
            SPLIT_SIZE (float): proportion of the dataset to be used for training

        Returns:
            None
        """
        
        categories = ['Cyst', 'Normal', 'Stone', 'Tumor']
        SPLIT_SIZE = self.config.split_ratio
        
        for category in categories:
            SOURCE_DIR = os.path.join(self.config.source_path, category)
            TRAINING_DIR = os.path.join(self.train_path, category)
            VALIDATION_DIR = os.path.join(self.val_path, category)
            
            
            # Check if the directories exist; if not, create them
            if not os.path.exists(TRAINING_DIR):
                os.makedirs(TRAINING_DIR)
            if not os.path.exists(VALIDATION_DIR):
                os.makedirs(VALIDATION_DIR)

            # Get the list of files
            files = os.listdir(SOURCE_DIR)

            # Shuffle the list of files
            random.sample(files, len(files))

            # Calculate the split index based on SPLIT_SIZE
            split_index = int(SPLIT_SIZE * len(files))

            # Separate files into training and validation sets
            training_files = files[:split_index]
            validation_files = files[split_index:]

            # Copy files to training directory
            for file in training_files:
                source = os.path.join(SOURCE_DIR, file)
                destination = os.path.join(TRAINING_DIR, file)
                if os.path.getsize(source) > 0:
                    copyfile(source, destination)
                else:
                    logger.info(f"{file} is zero leng, th ignoring.")

            # Copy files to validation directory
            for file in validation_files:
                source = os.path.join(SOURCE_DIR, file)
                destination = os.path.join(VALIDATION_DIR, file)
                if os.path.getsize(source) > 0:
                    copyfile(source, destination)
                else:
                    logger.info(f"{file} is zero length, so ignoring.")
