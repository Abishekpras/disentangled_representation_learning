import os
DATA_FOLDER = 'data/'
DSPRITES_FILENAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
MNIST_FILENAME = 'mnist_train_data.npz'

config = {'data_folder' : os.path.join(os.getcwd(), DATA_FOLDER),

		  'data_path' : {'dsprites' : os.path.join(os.getcwd(),
		  										   DATA_FOLDER,
		  										   DSPRITES_FILENAME),
		  				 'mnist' : os.path.join(os.getcwd(),
		  				 						DATA_FOLDER,
		  				 						MNIST_FILENAME)
		  				},
		  'img_dim' : 64
		 }