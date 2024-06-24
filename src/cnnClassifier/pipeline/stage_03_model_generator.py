from cnnClassifier import logger
from cnnClassifier.components.model_generation import ModelGeneration
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Generation"

class ModelGenerationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        modelprep_config = config.get_modelprep_config()
        model_generation = ModelGeneration(config=modelprep_config)
        model_generation.alexnet_with_improvements()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelGenerationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e