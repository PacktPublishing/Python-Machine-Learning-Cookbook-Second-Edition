from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

BasicModel=MobileNet(input_shape=(224, 224, 3), weights='imagenet',include_top=False)

ModelLayers=BasicModel.output
ModelLayers=GlobalAveragePooling2D()(ModelLayers)
ModelLayers=Dense(1024,activation='relu')(ModelLayers) 
ModelLayers=Dense(1024,activation='relu')(ModelLayers)
ModelLayers=Dense(512,activation='relu')(ModelLayers)
OutpModel=Dense(3,activation='softmax')(ModelLayers) 

ConvModel=Model(inputs=BasicModel.input,outputs=OutpModel)

for layer in ConvModel.layers[:20]:
    layer.trainable=False
for layer in ConvModel.layers[20:]:
    layer.trainable=True
    
TrainDataGen=ImageDataGenerator(preprocessing_function=preprocess_input) 

TrainGenerator=TrainDataGen.flow_from_directory('training_images/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

ConvModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


StepSizeTrain=TrainGenerator.n//TrainGenerator.batch_size
ConvModel.fit_generator(generator=TrainGenerator,
                   steps_per_epoch=StepSizeTrain,
                   epochs=10)
