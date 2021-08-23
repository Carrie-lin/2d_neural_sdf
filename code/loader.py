import torch
from torch.utils.data import Dataset


class SDFData(Dataset):
    def __init__(self, file):
        """
        :param file: Should be in the format of "dir/file.txt"
        """
        self.file = file
        self.data = self.load()

    def __getitem__(self, item):
        xy = self.data[item, :2]
        sdfxy = self.data[item, 2:]
        #print(sdfxy)
        return xy, sdfxy

    def __len__(self):
        return len(self.data)

    def load(self):
        data = []
        f = open(self.file, 'r')
        line = f.readline()
        while line:
            x, y, sdfx,sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
            data.append([x, y, sdfx,sdfy])
            line = f.readline()
        f.close()
        return torch.Tensor(data)
