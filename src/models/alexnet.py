from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization
)
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from models.abstract.model import Model
from models.utils import step_decay


_IMG_SIZE = 227
_EPOCHS = 20
_BATCH_SIZE = 20


class AlexNet(Model):
    def __init__(self):
        super().__init__("AlexNet", _IMG_SIZE, _EPOCHS, _BATCH_SIZE)

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(4, 4), activation="relu", padding="same", input_shape=(self.img_size, self.img_size, 3)))
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        Dropout(0.25),
        model.add(Conv2D(256, (5, 5), strides=(2, 2), activation="relu", padding="same"))
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        Dropout(0.5),
        model.add(Dense(4096, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model


    def get_model_callbacks(self) -> list:
        learning_rate_scheduler = LearningRateScheduler(step_decay)
        reduce_learning_rate_on_plateu = ReduceLROnPlateau(monitor="val_loss", patience=1)
        early_stopping = EarlyStopping(monitor="val_loss", patience=2)

        return [learning_rate_scheduler, reduce_learning_rate_on_plateu, early_stopping]
