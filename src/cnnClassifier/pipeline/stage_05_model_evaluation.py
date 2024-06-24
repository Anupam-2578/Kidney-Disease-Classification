from cnnClassifier import logger
from cnnClassifier.components.evaluation import ModelEvaluation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    obj = ModelEvaluation(config=evaluation_config)
    obj.evaluate_model()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e: 
    logger.exception(e)
    raise e