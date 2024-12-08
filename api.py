from fastapi import FastAPI,UploadFile,File
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from werkzeug.utils import secure_filename
import pickle


api=FastAPI()
model =load_model('BrainTumor10Epochs.h5')


	
# def get_result(img):
#     image=Image.open(img.file)
#     image = Image.fromarray(image, 'RGB')
#     image = image.resize((64, 64))
#     image=np.array(image)
#     input_img = np.expand_dims(image, axis=0)
#     return input_img

@api.get("/")
def read_root():
    return {"Hello": "World"}

@api.post("/predict")
def upload_file(my_img: UploadFile = File(...)):
    # img=Image.open(my_img.file)
    # result=get_result(my_img)
    image=Image.open(my_img.file)
    image = image.convert('RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    prediction = result.tolist() 
    return {"prediction": prediction}