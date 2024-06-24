from cnnClassifier import logger 
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



class PredictionPipeline:
    def __init__(self,filename:str):
        self.filename = filename
        

    def main(self):
        # Load the image
        image_path = os.path.join(self.filename)
        model = load_model(os.path.join("artifacts","model", "model.h5"))
        img = load_img(image_path, target_size=(512, 512))

        # Convert the Image to a NumPy array and normalize pixel values
        x = img_to_array(img)
        x = x / 255.0  # Normalize pixel values to the range [0, 1]

        # Expand the dimensions to match the expected input shape of the model
        x = np.expand_dims(x, axis=0)

        # Make predictions using the model
        predictions = model.predict(x)

        # Interpret the predictions
        class_labels = ["Cyst", "Normal", "Stone", "Tumor"]
        label_index = np.argmax(predictions[0])
        label = class_labels[label_index]

        # Plot the image
        plt.imshow(img)
        plt.title("Predicted Label: " + label)