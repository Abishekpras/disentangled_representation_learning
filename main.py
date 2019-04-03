import os
import logging
import argparse

import numpy as np
import torch
import torchvision

from config import config
from models import beta_vae, vae
from train import train_vae, train_beta_vae

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
parser.add_argument('--model', type=str, default='vae',
					 help='(Str) : vae || beta_vae')
args = parser.parse_args()

def run():

	if args.cuda and not torch.cuda.is_available():
		logging.warn('CUDA NOT AVAILABLE. CONTINUING WITH CPU PROCESSING')
		args.cuda = False

	device = torch.device('cuda' if args.cuda else 'cpu')

	if(args.model == 'beta_vae'):
		model = beta_vae()
		model = train_beta_vae(model, config['batch_size'])
	elif(args.model=='vae'):
		model = vae()
		model = train_vae(model, config['batch_size'])

if __name__ == '__main__':
	torch.multiprocessing.freeze_support()
	run()





