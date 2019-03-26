import torch

from torch.utils.data import Dataset, DataLoader
from config import config

class dspites_dataset(Dataset):
		def __init__(self,data):
			self.data = torch.tensor(data).type(torch.FloatTensor)

		def __len__(self):
			return len(self.data)

		def __getitem__(self, ix):
			return self.data[ix].view(-1, 1)

def create_data_loader(data, batch_size):

	dset = dspites_dataset(data)
	data_loader = DataLoader(dset, batch_size=batch_size,
							 shuffle=True, drop_last=True)
	return data_loader
