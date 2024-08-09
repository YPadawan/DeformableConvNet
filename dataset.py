import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WaferDefaultDataset(Dataset):
    
    def __init__(self, dataset_dir, train, test_size=0.2):
        """
        dataset_dir : 'str', path to the directory of the wafer default dataset
        train: 'bool' a boolean used for splitting the of train and test datasets, if true it will return the
                training set, if False, it will return the test set
        test_size: 'float' a float between 0 and 1 (default 0.2) for splitting the dataset between train and test sets
        """
        
        self.data = np.load(dataset_dir)
        self.fullsize = len(self.data["arr_1"])
        self.train = train # boolean used to split a train set from a test set
        self.split_idx = int(self.fullsize * (1-test_size))
        
        if self.train:
            
            self.datasamples = self.data["arr_0"][:self.split_idx]
            self.labels = self.data["arr_1"][:self.split_idx]
        
        else:
            
            self.datasamples = self.data["arr_0"][self.split_idx:]
            self.labels = self.data["arr_1"][self.split_idx:]
            

    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        sample = self.datasamples[idx]
        label = self.labels[idx]
        
        return torch.from_numpy(sample.astype('float32')).unsqueeze(0), torch.from_numpy(label.astype('float32')).unsqueeze(0)