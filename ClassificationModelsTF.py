import enum

from keras.layers import Conv2D, Input,MaxPooling2D, Reshape,Activation,Flatten, Dense,Dropout
from keras.models import Model, Sequential

class ClassifierType(enum.Enum):
    Simple1 = 1
    EfficentNetB0 = 2

class ClassificationModelGTDetection:

    @staticmethod
    def getSimpleModel(size , classes_count):

        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=1, input_shape=(size[0], size[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), input_shape=(size[0], size[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), input_shape=(size[0], size[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Reshape((-1,)))

        model.add(Dense(units=1000, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(units=500, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(units=100, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(units=classes_count, activation='softmax'))
        return model

    @staticmethod
    def getModel(type, size, classes_count):
        if type == ClassifierType.Simple1:
           return ClassificationModelsTF.getSimpleModel(size, classes_count)
