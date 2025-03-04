from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from read_feature import encode_feature

class ImageDataset(Dataset):
    def __init__(self, dataPath:str):
        self.dataPath = dataPath
        self.originDatas = encode_feature(dataPath)
        self.header = self.originDatas[0]
        self.datas = self.originDatas[1:]
        self.total = len(self.datas)
        self.dataNumpy = self.datas
        # print(self.dataDf)


    def __getitem__(self, index):
        hot = [0,0,0]
        hot[int(self.dataNumpy[index][0])] = 1
        return self.dataNumpy[index][0], np.array(hot), self.dataNumpy[index][1], np.array(self.dataNumpy[index][2:], dtype=np.float64)


    def __len__(self):
        return self.total