import torch


class BenchTempOptimizer:
    def __init__(self, model, lr, optimizer="adam"):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def adam(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
