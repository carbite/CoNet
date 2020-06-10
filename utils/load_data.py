import numpy as np
import torch

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path):

        self.X = np.array(np.load(x_path))
        self.Y = np.array(np.load(y_path))
    def __getitem__(self, index):

        return self.X[index], self.Y[index]
    def getall(self):

        return self.X, self.Y
    def __len__(self):

        return len(self.X)