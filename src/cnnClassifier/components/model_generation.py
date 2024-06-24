import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from cnnClassifier.entity.config_entity import DatapreparationConfig
from cnnClassifier import logger



class ModelGeneration:
    def __init__(self , config : DatapreparationConfig):
        self.config = config
    
    def alexnet_with_improvements(self)->None:
        input_shape = self.config.params_image_size
        num_classes = self.config.params_classes   
         
        model = tf.keras.models.Sequential([
            Conv2D(96, (11, 11), strides=(4, 4), padding='same', input_shape=input_shape),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),  # Using LeakyReLU instead of ReLU
            MaxPooling2D((3, 3), strides=(2, 2)),

            Conv2D(256, (5, 5), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D((3, 3), strides=(2, 2)),

            Conv2D(384, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),

            Conv2D(384, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),

            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPooling2D((3, 3), strides=(2, 2)),

            Flatten(),

            Dense(1024),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),  # Dropout for regularization

            Dense(1024, kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),

            Dense(1024, kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),

            Dense(num_classes, activation='softmax')
        ])

        # Print model summary
        model.summary()

        # Define the optimizer with specific parameters
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,  # Learning rate
            rho=0.9,              # Decay rate for the moving average of square gradients
            momentum=0.0,          # Momentum parameter
            epsilon=1e-07,         # A small constant for numerical stability
            centered=False        # If True, gradients are normalized by the estimated variance of the gradient
        )

        # Compile the model with specified loss function and evaluation metric
        model.compile(optimizer=optimizer,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
        
        model.save(self.config.model_path)
        
        logger.info(f"Model saved at {self.config.model_path}")