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

_IMG_SIZE = 224
_EPOCHS = 20


class VGG16(Model):
    def __init__(self):
        super().__init__(_IMG_SIZE, _EPOCHS)

    def create_model(self):
        model = Sequential([
            Model.data_augmentation(),
            Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(self.img_size, self.img_size, 3)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Dropout(0.25),
            Conv2D(256, (2, 2), activation="relu", padding="same"),
            Conv2D(256, (2, 2), activation="relu", padding="same"),
            Conv2D(256, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Conv2D(512, (2, 2), activation="relu", padding="same"),
            Conv2D(512, (2, 2), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Dropout(0.25),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Conv2D(512, (2, 2), activation="relu", padding="same"),
            Conv2D(512, (2, 2), activation="relu", padding="same"),
            Conv2D(512, (2, 2), activation="relu", padding="same"),
            MaxPooling2D((2, 2), strides=(2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.25),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def get_model_callbacks(self) -> list:
        return []
