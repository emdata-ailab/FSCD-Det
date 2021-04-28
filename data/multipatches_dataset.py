import numpy as np
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util
import torch
from torchvision import transforms



def unnormalize(tensor, mean, std):
	for t, m, s in zip(tensor, mean, std):
		t.mul_(s).add_(m)
	return tensor

class MultiPatchesDataset(BaseDataset):
	"""
	This dataset class can load unaligned/unpaired datasets.

	It requires two directories to host training images from domain A '/path/to/data/trainA'
	and from domain B '/path/to/data/trainB' respectively.
	You can train the model with the dataset flag '--dataroot /path/to/data'.
	Similarly, you need to prepare two directories:
	'/path/to/data/testA' and '/path/to/data/testB' during test time.
	"""

	def __init__(self, opt):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)

		self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
		self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

		if opt.phase == 'test' and not os.path.exists(self.dir_A) \
			and os.path.exists(os.path.join(opt.dataroot, 'valA')):
			self.dir_A = os.path.join(opt.dataroot, 'valA')
			self.dir_B = os.path.join(opt.dataroot, 'valB')

		self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size)) # load images from '/path/to/data/trainA'
		self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size)) # load images from '/path/to/data/trainB'
		self.A_size = len(self.A_paths) # get the size of dataset A
		self.B_size = len(self.B_paths) # get the size of dataset B

		# In single-image translation, we augment the data loader by applying
		# random scaling. Still, we design the data loader such that the
		# amount of scaling is the same within a minibatch. To do this,
		# we precompute the random scaling values, and repeat them by |batch_size|.
		# A_zoom = 1 / self.opt.random_scale_max
		# zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
		# self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])

		# B_zoom = 1 / self.opt.random_scale_max
		# zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
		# self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index (int)      -- a random integer for data indexing

		Returns a dictionary that contains A, B, A_paths and B_paths
			A (tensor)       -- an image in the input domain
			B (tensor)       -- its corresponding image in the target domain
			A_paths (str)    -- image paths
			B_paths (str)    -- image paths
		"""
		A_path = self.A_paths[index % self.A_size]
		if self.opt.serial_batches: # make sure index is within the range
			index_B = index % self.B_size
		else: # randomize the index for domain B to avoid fixed pairs.
			index_B = random.randint(0, self.B_size - 1)
		B_path = self.B_paths[index_B]
		A_img = Image.open(A_path).convert('RGB')
		B_img = Image.open(B_path).convert('RGB')
		# apply image transformation
		is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
		modified_opt = util.util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
		
		# the dataloader pipeline does not change for B domain
		transform = get_transform(modified_opt)
		A = transform(A_img)
		B = transform(B_img)

		data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

		# import pdb;pdb.set_trace()
		# extend with different ratio
		A_extend = []
		ratios = self.opt.ratios
		if ratios is None:
			return data
		else:
			ratios = [[1.0/ratios[i], 1.0/ratios[i+1]] for i in range(0,len(ratios), 2) ]

		for ratio_w, ratio_h in ratios:		
			A_copy = A.clone()
			# Different Ratio
			trans = transforms.Compose([transforms.Lambda(lambda img: unnormalize(img, (0.5,0.5,0.5), (0.5,0.5,0.5))),
																		transforms.ToPILImage(), 
																		transforms.Resize([int(A.shape[1]*ratio_w), int(A.shape[2]*ratio_h)], interpolation=Image.NEAREST),
																		transforms.ToTensor(),
																		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
																	])
			A_extend.append(trans(A_copy))
		data['A_extend'] = A_extend
		
		return data 

	def __len__(self):
		""" Return the total number of images in the dataset.

		As we have two datasets with potentially different number of images.
		we take a maximum of 
		"""
		return max(self.A_size, self.B_size)
