import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from torchvision.transforms import Compose,  ToTensor, Normalize
from PIL import Image
import random,cv2
import numpy as np
import util.util as util


class UnalignedDataset(BaseDataset):
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
        if opt.phase == "test":
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_img = np.array(A_img)
        B_img = np.array(B_img)

        # Apply image transformation
        if 'crop' in self.opt.preprocess:
            if self.opt.serial_batches:
                len_A_r = A_img.shape[0] - self.opt.crop_size
                len_A_c = A_img.shape[1] - self.opt.crop_size
                if len_A_r <= 0 or len_A_c < 0:
                    A_img = cv2.resieze(A_img,(self.opt.crop_size,self.opt.crop_size))
                else:
                    row = np.random.randint(len_A_r)
                    col = np.random.randint(len_A_c)
                    A_img = A_img[row:row + self.opt.crop_size, col:col + self.opt.crop_size, :]
                    B_img = B_img[row:row + self.opt.crop_size, col:col + self.opt.crop_size, :]
            else:
                len_A_r = A_img.shape[0] - self.opt.crop_size
                len_A_c = A_img.shape[1] - self.opt.crop_size
                if len_A_r <= 0 or len_A_c <= 0:
                    A_img = cv2.resize(A_img,(self.opt.crop_size,self.opt.crop_size))
                else:
                    row = np.random.randint(len_A_r)
                    col = np.random.randint(len_A_c)
                    A_img = A_img[row:row + self.opt.crop_size, col:col + self.opt.crop_size, :]
                len_B_r = B_img.shape[0] - self.opt.crop_size
                len_B_c = B_img.shape[1] - self.opt.crop_size
                if len_B_r <= 0 or len_B_c <= 0:
                    B_img = cv2.resize(B_img, (self.opt.crop_size, self.opt.crop_size))
                else:
                    row = np.random.randint(len_B_r)
                    col = np.random.randint(len_B_c)
                    B_img = B_img[row:row + self.opt.crop_size, col:col + self.opt.crop_size, :]
        A_img = self.transpose(A_img)
        B_img = self.transpose(B_img)
        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    def transpose(self, data):
        out = Image.fromarray(data)
        if self.transforms:
            out = self.transforms(out)
        return out
