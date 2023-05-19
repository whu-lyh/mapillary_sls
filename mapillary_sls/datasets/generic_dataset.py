#  Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from os.path import join

class ImagesFromList(Dataset):
	'''
		return images and idx from dataset, and there are some arguments, e.g. data argumentation.
	'''
	def __init__(self, images, transform):
        # the __init__ method of a dataset class is typically used for initializing the dataset object.
		# It is called when you create an instance of the dataset class.
	    self.images = np.asarray(images)
	    self.transform = transform

	def __len__(self):
	    return len(self.images)

	def __getitem__(self, idx):
		'''
			define the behavior for accessing individual elements or samples from the dataset.
		'''
		img = [Image.open(im) for im in self.images[idx].split(",")]
		img = [self.transform(im) for im in img]

		if len(img) == 1:
			img = img[0]
		#print("idx:",idx)
		return img, idx
