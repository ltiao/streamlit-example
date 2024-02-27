import torch


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_float():
    return torch.double
