import torch


class BenchTempLoss:
    def __init__(self, loss="bce"):
        self.loss = loss

    def loss(self):
        if self.loss == "bce":
            return torch.nn.BCELoss()
        return torch.nn.BCELoss()