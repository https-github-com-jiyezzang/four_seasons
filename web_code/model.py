from keras.models import load_model
from PIL import Image
import numpy as np

def predict(file_path):
    
    # TODO: Load the trained model
    model = load_model('/home/jiyezzangg/project/best_model.h5')
    
    # TODO: Preprocess the image
    img = Image.open(file_path)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = img_array.reshape(1,224,224,-1)
    
    # TODO: Use the model to predict the digit
    predict_value = model.predict(img_array)
    warm = predict_value[0][0]
    warm = int(warm * 100)
    cool = predict_value[0][1]
    cool = int(cool * 100)
    
    return warm, cool