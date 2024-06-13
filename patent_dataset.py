import json
import pandas as pd
from torch.utils.data import Dataset


class PatentDataset(Dataset):
    def __init__(self, dataset_path: str):
        
        # Loading dataset
        with open(dataset_path, "r") as json_file:
            dataset = json.load(json_file)
        self.df = pd.DataFrame(dataset)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx].tolist()  # query, negative, positive
