from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout
)
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from models.utils import step_decay
from models.abstract.model import Model

_IMG_SIZE = 200
_EPOCHS = 20


class CNN(Model):
    def __init__(self):
        super().__init__("CNN", _IMG_SIZE, _EPOCHS)

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Conv2D(16, (5, 5), activation="relu", padding="same", input_shape=(self.img_size, self.img_size, 3)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
        model.add(MaxPooling2D((3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model

    def get_model_callbacks(self) -> list:
        learning_rate_scheduler = LearningRateScheduler(step_decay)
        reduce_learning_rate_on_plateu = ReduceLROnPlateau(monitor="val_loss", patience=1)
        early_stopping = EarlyStopping(monitor="val_loss", patience=2)

        return [learning_rate_scheduler, reduce_learning_rate_on_plateu, early_stopping]
