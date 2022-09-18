from abc import ABC, abstractmethod
from typing import Tuple, Any, TypeVar
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import RandomFlip
from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from sklearn.model_selection import StratifiedKFold, train_test_split

ModelHistory = TypeVar("ModelHistory", list, dict)

_DEFAULT_IMG_SIZE = 250
_DEFAULT_EPOCHS = 20
_SIGMOID_BINARY_CLASS_THRESHOLD = 0.5


class Model(ABC):
    def __init__(
            self,
            img_size: int = _DEFAULT_IMG_SIZE,
            epochs: int = _DEFAULT_EPOCHS
    ):
        self.model = None
        self.img_size = img_size
        self.epochs = epochs

    @abstractmethod
    def create_model(self) -> Sequential:
        pass

    @abstractmethod
    def get_model_callbacks(self) -> list:
        pass

    def train_once(self, files_df) -> ModelHistory:
        train_df, test_df = train_test_split(files_df, test_size=0.2)
        train_data, test_data = self._get_training_test_data_generators(train_df, test_df)

        history = self._train(train_data, test_data)

        return history

    def cross_validate(self, files_df: pd.DataFrame, k: int = 5) -> Tuple[list, list]:
        kf = StratifiedKFold(n_splits=k, shuffle=True)

        histories = []
        losses = []
        fold = 0
        for train_idx, test_idx in kf.split(files_df["file"], files_df["label"]):
            print(f"Fold #{++fold + 1}")

            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)

            train = files_df.loc[train_idx]
            test = files_df.loc[test_idx]

            train_data, test_data = self._get_training_test_data_generators(train, test)

            history = self._train(train_data, test_data)
            histories.append(history)

            zero_one_loss = self.test_model(test_data)
            losses.append(np.mean(zero_one_loss))

        return histories, losses

    def test_model(self, test_data: DataFrameIterator) -> list:
        x_test, y_test = self._get_set_to_test_model(test_data)
        pred = self.model.predict(x_test)

        pred_classes = [0 if p < _SIGMOID_BINARY_CLASS_THRESHOLD else 1 for p in pred]
        loss = [0 if int(pred_class) == int(true_val) else 1 for (pred_class, true_val) in zip(pred_classes, y_test)]

        return loss

    def _train(
            self,
            train_data: DataFrameIterator,
            test_data: DataFrameIterator
    ) -> ModelHistory:
        self.model = self.create_model()

        history = self.model.fit(
            train_data,
            validation_data=test_data,
            epochs=self.epochs,
            callbacks=self.get_model_callbacks()
        )

        return history

    def _get_set_to_test_model(self, test_data: DataFrameIterator) -> tuple:
        test_data_set = next(test_data)
        x_test = test_data_set[0]
        y_test = test_data_set[1]

        return x_test, y_test

    def _get_training_test_data_generators(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame
    ) -> Tuple[DataFrameIterator, DataFrameIterator]:
        train_data_generator = ImageDataGenerator(rescale=1. / 255)
        test_data_generator = ImageDataGenerator(rescale=1. / 255)

        train_data = train_data_generator.flow_from_dataframe(train_data, x_col="file", y_col="label", color_mode="rgb",
                                                              target_size=(self.img_size, self.img_size),
                                                              class_mode="binary",
                                                              shuffle=True)
        test_data = test_data_generator.flow_from_dataframe(test_data, x_col="file", y_col="label", color_mode="rgb",
                                                            target_size=(self.img_size, self.img_size),
                                                            class_mode="binary")

        return train_data, test_data

    @staticmethod
    def data_augmentation() -> Sequential:
        return Sequential(
            [
                RandomFlip("horizontal"),
            ]
        )

