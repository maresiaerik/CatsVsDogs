from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization
)
from models.abstract.model import Model
from keras.callbacks import EarlyStopping

_IMG_SIZE = 224
_EPOCHS = 20


class VGG16(Model):
    def __init__(self):
        super().__init__("VGG16", _IMG_SIZE, _EPOCHS)

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(self.img_size, self.img_size, 3)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(256, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(256, (2, 2), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(512, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(512, (2, 2), activation="relu", padding="same"))
        model.add(Conv2D(512, (2, 2), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def get_model_callbacks(self) -> list:
        early_stopping = EarlyStopping(monitor="val_loss", patience=2)

        return [early_stopping]
