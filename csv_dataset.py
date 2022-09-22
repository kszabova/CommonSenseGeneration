from torch.utils.data import Dataset
import pandas as pd
import os


class CSVDataset(Dataset):
    def __init__(self, file):
        super().__init__()

        if not os.path.isfile(file):
            raise RuntimeError("The file does not exist")

        self.file = file
        self.df = pd.read_csv(self.file, header=None)
        self.length = len(self.df)

    def __getitem__(self, index):
        start, end, sentence = self.df.iloc[index]
        return {"concepts": [start, end], "target": sentence}

    def __len__(self):
        return self.length
