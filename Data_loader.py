import re
import torch
import os
import numpy as np
from torch.utils.data import Dataset

shuffle = True
batchSize = 16

class DataSet(Dataset):
    def __init__(self,folderName):
        self.folderPath = folderName
        self.indexList = []
        self.globalIndex = 0 #only if shuffle is off
        self.determineIndices()

    def determineIndices(self):
        for fileName in os.listdir(self.folderPath):  
            number = int(re.search(r'\d+', fileName).group())
            if number not in self.indexList:
                self.indexList.append(number)

    def __len__(self):
        return len(self.indexList)

    def __getitem__(self, idx):
        index_string = str(self.indexList[idx])
        blurredVolume = np.load(self.folderPath + "/blurred_subject_" + index_string + ".npy")
        originalVolume = np.load(self.folderPath + "/subject_" + index_string + ".npy")

        blurred_tensor = torch.from_numpy(blurredVolume).float().unsqueeze(0)
        sharp_tensor = torch.from_numpy(originalVolume).float().unsqueeze(0)
        
        return blurred_tensor,sharp_tensor


    