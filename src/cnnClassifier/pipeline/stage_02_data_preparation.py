from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preparation import DataPreparation
from cnnClassifier import logger

STAGE_NAME = "Data Preparation"

class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.create_train_val_dirs()
        data_preparation.split_data()
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e