from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout
)

from models.abstract.model import Model

_IMG_SIZE = 200
_EPOCHS = 20


class CNN(Model):
    def __init__(self):
        super().__init__(_IMG_SIZE, _EPOCHS)

    def create_model(self) -> Sequential:
        model = Sequential([
            Model.data_augmentation(),
            Conv2D(16, (5, 5), activation="relu", padding="same", input_shape=(self.img_size, self.img_size, 3)),
            MaxPooling2D((3, 3)),
            Conv2D(32, (5, 5), activation="relu", padding="same"),
            MaxPooling2D((3, 3)),
            Conv2D(32, (5, 5), activation="relu", padding="same"),
            MaxPooling2D((3, 3)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model

    def get_model_callbacks(self) -> list:
        return []
