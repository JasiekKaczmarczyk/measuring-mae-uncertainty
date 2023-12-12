import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

class MAEDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset
        # self.processor = processor

        self.key = "image" if "image" in self.dataset.features else "img"

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        record = self.dataset[index]
        image = record[self.key]

        return np.array(image)