import os
import logging
import argparse

import numpy as np
import torch
import torchvision

from config import config
from data_manager import create_data_loader
from models import beta_vae


parser = argparse.ArgumentParser(description='Beta-VAE Example implementation')
parser.add_argument('--batch-size', type=int, default=256,
					 help='(Int) : Number of images in a single batch input')
parser.add_argument('--epochs', type=int, default=25,
					 help='(Int) : Number of times whole dataset is scanned\
					 during training')
parser.add_argument('--cuda', type=bool, default=False,
					 help='(Bool) : True => GPU Training, False => CPU')
parser.add_argument('--dataset', type=str, default='mnist',
					 help='(Str) : mnist || dsprites')
args = parser.parse_args()

def run():

	if args.cuda and not torch.cuda.is_available():
		logging.warn('CUDA NOT AVAILABLE. CONTINUING WITH CPU PROCESSING')
		args.cuda = False

	device = torch.device('cuda' if args.cuda else 'cpu')

	data = np.load(config['data_path'][args.dataset], encoding='bytes')

	data_loader = create_data_loader(data=data['imgs'],
									 batch_size=args.batch_size,
									 dset_name=args.dataset)

	D_in = D_out = config['img_dim']**2
	H = 100

	model = beta_vae(D_in, H, D_out)
	'''
	for x in iter(data_loader):
		pass
	'''

if __name__ == '__main__':
	torch.multiprocessing.freeze_support()
	run()





