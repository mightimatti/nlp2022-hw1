from torch.utils.data import Dataset

# Based on QnA Notebook
class NERDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> List[Dict]:
        return self.data[index]
