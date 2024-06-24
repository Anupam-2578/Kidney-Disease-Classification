from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig , DatapreparationConfig , ModelprepConfig,ModelTrainingConfig,EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preparation_config(self)-> DatapreparationConfig:
        config = self.config.data_preparation

        create_directories([config.root_dir])

        data_preparation_config = DatapreparationConfig(
            root_dir=config.root_dir,
            source_path=config.data_path,
            train_path=config.train_path,
            val_path=config.val_path,
            split_ratio=config.split_ratio
        )

        return data_preparation_config
    
    def get_modelprep_config(self) -> ModelprepConfig:
        config = self.config.prepare_model

        create_directories([config.root_dir])

        modelprep_config = ModelprepConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES
        )

        return modelprep_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            untrained_model_path=config.untrained_model_path,
            trained_model_path=config.trained_model_path,
            training_directory=config.training_directory,
            validation_directory=config.validation_directory
        )

        return model_training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation

        create_directories([config.root_dir])

        evaluation_config = EvaluationConfig(
            root_dir=config.root_dir,
            model_history_path= config.model_history_path
        )

        return evaluation_config