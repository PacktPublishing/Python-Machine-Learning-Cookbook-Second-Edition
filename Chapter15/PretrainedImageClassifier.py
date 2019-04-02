from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

PTModel = ResNet50(weights='imagenet')

ImgPath = 'airplane.jpg'
Img = image.load_img(ImgPath, target_size=(224, 224))
InputIMG = image.img_to_array(Img)
InputIMG = np.expand_dims(InputIMG, axis=0)
InputIMG = preprocess_input(InputIMG)

PredData = PTModel.predict(InputIMG)

print('Predicted:', decode_predictions(PredData, top=3)[0])