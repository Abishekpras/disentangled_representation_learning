import torch
import logging
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from config import config

class custom_dataset(Dataset):
		def __init__(self, data, transform=None):
			self.data = torch.tensor(data).type(torch.FloatTensor)
			self.transform = transform

		def __len__(self):
			return len(self.data)

		def __getitem__(self, ix):
			if self.transform:
				pil_img = lambda x: F.to_pil_image(np.uint8(x))
				self.transform(pil_img(self.data[ix]))
			return self.data[ix].view(-1, 1)

def create_data_loader(data, batch_size, dset_name='dsprites'):

	data_transform = None
	if dset_name == 'mnist':
		logging.warn('TESTING WITH MNIST DATA')
		data_transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))])

	dset = custom_dataset(data, data_transform)
	data_loader = DataLoader(dset, batch_size=batch_size,
							 shuffle=True, drop_last=True)
	return data_loader
