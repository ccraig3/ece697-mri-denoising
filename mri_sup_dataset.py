# MRI Supervised Dataset
# Written by: Cameron Craig
from unet import UNet
import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms, _utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import pickle as pkl


###   HELPER FUNCTIONS   ###
# Obtain a sample of a uniform random variable on the specified bounds
def sampleRV(BOUNDS):
  return np.random.uniform(BOUNDS[0], BOUNDS[1])

def clip_img(image):
  image = np.where(image > 1, 1, image)
  return np.where(image < 0, 0, image)

def normalize(image):
  new_image = image - np.min(image)
  return new_image / np.max(new_image)

def add_rician_noise(image, gain=0.1):
  image = image.numpy()#.resize((image.shape[-2], image.shape[-1]))
  n1 = normalize(np.random.normal(0, 1, (image.shape[-2], image.shape[-1])))
  n2 = normalize(np.random.normal(0, 1, (image.shape[-2], image.shape[-1])))

  return torch.tensor(clip_img(np.abs(image + gain*n1 + gain*n2*1j)))

def noiseSimV1(img_tensor):
  img_tensor = img_tensor.detach().cpu()
  SIZE = img_tensor.shape[-1]
  x = np.linspace(-1, 1, num=SIZE)

  # Vertical
  y_vert = (1 / (0.8 + np.exp(-3*x))) + 0.2
  vert_mask = (normalize(np.tile(np.reshape(y_vert, (-1, 1)), (1, SIZE))) * 0.95) + 0.1

  # Horizontal
  y_horiz = -(x**4)
  y_horiz = (y_horiz / np.max(np.absolute(y_horiz))) + 1
  horiz_mask = normalize(np.tile(np.reshape(y_horiz, (1, -1)), (SIZE, 1))) + 0.05

  # Combined
  mask = torch.tensor(normalize(4*vert_mask * horiz_mask))

  noise_intensity = random.random()*0.05 + 0.05 # 0.05 to 0.10, uniformly distributed
  return add_rician_noise(torch.mul(img_tensor, mask), noise_intensity)

def noiseSimV2(img_tensor, bias_tensor, noise_gain=0.1):
  img_tensor = img_tensor.detach().cpu()
  SIZE = img_tensor.shape[-1] # TODO Remove if possible

  if img_tensor.shape != bias_tensor.shape:
    print('ERROR: noiseSimV2: args img_tensor and bias_tensor have different shapes')
    print('  -> img_tensor.shape = ' + str(img_tensor.shape))
    print('  -> bias_tensor.shape = ' + str(bias_tensor.shape))
  
  biased_tensor = (img_tensor * bias_tensor)# * 255
  return add_rician_noise(biased_tensor, gain=noise_gain)


###   CORE DATASET CLASS   ###
class MriSupDataset(Dataset):
    """Mri supervised dataset."""

    def __init__(self, root_dir, bias_fields, noise_bounds, size=512, preload=False):
        super().__init__()
        """
        Args:
            root_dir      (string):         Path of directory with all the image files
            bias_fields   ([np.ndarray]):   List of bias field numpy arrays
            noise_bounds  ((LOWER, UPPER)): Tuple of lower and upper bounds for noise gain
            size          (int):            Square image side length in pixels
            preload       (boolean):        Load all images in root_dir into memory on init
        """
        self.root_dir = root_dir
        self.file_names = [name for name in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, name))]
        self.length = len(self.file_names)
        self.bias_fields = bias_fields

        # Make sure the dimensions of the bias fields match the images
        if bias_fields[0].shape[-1] != size:
          for i in range(len(bias_fields)):
            self.bias_fields[i] = cv2.resize(bias_fields[i], dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        
        # Fix any NaNs in bias fields
        for i in range(len(bias_fields)):
          np.nan_to_num(self.bias_fields[i], copy=False, nan=0.0, posinf=1.0, neginf=0.0)

        self.noise_bounds = noise_bounds
        self.SIZE = size
        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize((self.SIZE, self.SIZE))
        ])
        
        self.preload = preload
        if not preload:
          return
        
        # Load images into memory
        self.images = []
        for idx in range(self.length):
          img_path = os.path.join(self.root_dir, self.file_names[idx])
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
          img_tensor = self.transform(img).float()
          img_tensor -= img_tensor.min()
          img_tensor /= (img_tensor.max() + 1e-9)

          self.images.append(img_tensor)

    def __len__(self):
      return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.preload:
          img_tensor = self.images[idx]
        else:
          img_path = os.path.join(self.root_dir, self.file_names[idx])
          if os.path.exists(img_path):
            #img = im.open(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
          else:
            print('ERROR: __getitem__: Image index ' + str(idx) + ' was not found')
            print('  -> Missing file path: ' + str(img_path))

          img_tensor = self.transform(img).float()
          img_tensor -= img_tensor.min()
          img_tensor /= (img_tensor.max() + 1e-9)

        bias_tensor = torch.tensor(random.choice(self.bias_fields)).view(1, self.SIZE, self.SIZE)
        noise_gain = sampleRV(self.noise_bounds)

        sim_singlecoil_tensor = noiseSimV2(img_tensor, bias_tensor, noise_gain=noise_gain).float()
        sim_singlecoil_tensor -= sim_singlecoil_tensor.min()
        sim_singlecoil_tensor /= (sim_singlecoil_tensor.max() + 1e-9)

        return sim_singlecoil_tensor, img_tensor