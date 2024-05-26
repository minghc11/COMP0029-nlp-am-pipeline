import torch
from torch.utils.data import Dataset

class ComponentDataset(Dataset):
    def __init__(self, argument1s, argument2s, labels, tokenizer, max_length):
        self.argument1s = argument1s
        self.argument2s = argument2s
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.argument1s)
    def __getitem__(self, idx):
        argument1s = self.argument1s[idx]
        argument2s = self.argument2s[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            argument1s,
            argument2s,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}