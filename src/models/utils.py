import math


def step_decay(epoch: int) -> float:
    initial_learning_rate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return learning_rate


def exponential_decay(epoch: int) -> float:
    initial_rate = 0.1
    k = 0.1
    learning_rate = initial_rate * math.exp(-k*epoch)

    return learning_rate
