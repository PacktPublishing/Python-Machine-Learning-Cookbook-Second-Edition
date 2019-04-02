from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans

model = VGG16(weights='imagenet', include_top=False)

VGG16FeatureList = []


import os
for path, subdirs, files in os.walk('training_images'):
    for name in files:
        img_path = os.path.join(path, name)
        print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        VGG16Feature = model.predict(img_data)
        VGG16FeatureNp = np.array(VGG16Feature)
        VGG16FeatureList.append(VGG16FeatureNp.flatten())
        
VGG16FeatureListNp = np.array(VGG16FeatureList)

KmeansModel = KMeans(n_clusters=3, random_state=0)
KmeansModel.fit(VGG16FeatureListNp)

print(KmeansModel.labels_)