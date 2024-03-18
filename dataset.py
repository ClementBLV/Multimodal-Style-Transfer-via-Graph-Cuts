import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image

trans = transforms.Compose([
	transforms.RandomCrop(256),
	transforms.ToTensor()
	])

class PreProcessDataset(Dataset):
	"""docstring for PreProcessDataset"""
	def __init__(self, content_dir, style_dir, transforms=trans):
		super().__init__()
		assert not("." in content_dir) , "Don't extentions in the path"
		assert not("." in style_dir) , "Don't put extentions in the path"
		content_dir_resized = content_dir + '_resized'
		style_dir_resized = style_dir + '_resized'
		if not (os.path.exists(content_dir_resized) and os.path.exists(style_dir_resized)):
			os.mkdir(content_dir_resized)
			os.mkdir(style_dir_resized)
			self._resize(content_dir, content_dir_resized)
			self._resize(style_dir, style_dir_resized)

		content_images = glob.glob((content_dir_resized + '/*'))
		np.random.shuffle(content_images)
		style_images = glob.glob((style_dir_resized + '/*'))
		np.random.shuffle(style_images)
		self.image_pairs = list(zip(content_images, style_images))
		self.transforms = transforms

	@staticmethod
	def _resize(source_dir, target_dir):
		print(f'Start Resizing {source_dir} ')
		for i in tqdm(os.listdir(source_dir)):
			filename = os.path.basename(i)
			print(filename)
			try:
				image = io.imread(os.path.join(source_dir, i))
				if len(image.shape) == 3 and image.shape[-1] == 3:
					H, W, _ = image.shape

					if H < W:
						print('H < W')
						ratio = W / H
						H = 512
						W = int(ratio*H)
					else:
						ratio = H / W
						W = 512
						H = int(ratio*W)
					image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)

					# Convert image data type to uint8 if not it doesn't work
					image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

					io.imsave(os.path.join(target_dir, filename), image_uint8)
			except Exception as err:
				print(Exception, err)
				print("there were an issue during resize")
				continue

	def __len__(self):
		return len(self.image_pairs)

	def __getitem__(self, index):
		content_image, style_image = self.image_pairs[index]
		content_image = Image.open(content_image)
		style_image = Image.open(style_image)

		if self.transforms:
			content_image = self.transforms(content_image)
			style_image = self.transforms(style_image)
			return content_image, style_image

if __name__ == '__main__':
	cli = argparse.ArgumentParser(description='PreProcess Dataset to get transformed images')
	cli.add_argument('--train_content_dir', type=str, default=None,
					help='content images dir to train on')
	cli.add_argument('--train_style_dir', type=str, default=None,
					help='style images dir to train on')
	cli.add_argument('--test_content_dir', type=str, default=None,
					help='content images to test on')
	cli.add_argument('--test_style_dir', type=str, default=None,
					help='style images to test on')
	args = cli.parse_args()
	if args.train_content_dir is not None :
		PreProcessDataset(args.train_content_dir, args.train_style_dir)
	if args.test_content_dir is not None :
		PreProcessDataset(args.test_content_dir, args.test_style_dir)
