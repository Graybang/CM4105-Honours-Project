import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import functional as F


class Flickr1024(Dataset):
    def __init__(self, root_dir, im_size, scale, transform=None):

        self.root_dir = root_dir
        self.im_size = im_size
        self.scale = scale
        self.transform = transform

        images_L = []
        images_R = []
        for file in os.listdir(self.root_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            if 'l' in file.lower():
                images_L.append(file)
            else:
                images_R.append(file)
        
        images_L.sort()
        images_R.sort()

        self.images_L = images_L
        self.images_R = images_R
        self.labels_L = images_L
        self.labels_R = images_R

        print(len(images_L))

    def __len__(self):

        return len(self.images_L)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path_L = os.path.join(self.root_dir, self.images_L[idx])
        img_path_R = os.path.join(self.root_dir, self.images_R[idx])
        # label_path = os.path.join(self.root_dir + '/label', self.labels[idx])

        img_L = Image.open(img_path_L)
        img_L = img_L.resize((int(self.im_size / self.scale), int(self.im_size / self.scale)))
        img_R = Image.open(img_path_R)
        img_R = img_R.resize((int(self.im_size / self.scale), int(self.im_size / self.scale)))
        label_L = Image.open(img_path_L)
        label_L = label_L.resize((int(self.im_size), int(self.im_size)))
        label_R = Image.open(img_path_R)
        label_R = label_R.resize((int(self.im_size), int(self.im_size)))

        if self.transform:
            img_L, img_R, label_L, label_R = self.transform(img_L, img_R, label_L, label_R)
            # label = self.transform(label)

        return img_L, img_R, label_L, label_R


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_L, img_R, label_L, label_R):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img_L.transpose(Image.FLIP_LEFT_RIGHT), img_R.transpose(Image.FLIP_LEFT_RIGHT), label_L.transpose(Image.FLIP_LEFT_RIGHT), label_R.transpose(Image.FLIP_LEFT_RIGHT)
        return img_L, img_R, label_L, label_R


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_L, img_R, label_L, label_R):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img_L.transpose(Image.FLIP_TOP_BOTTOM), img_R.transpose(Image.FLIP_TOP_BOTTOM), label_L.transpose(Image.FLIP_TOP_BOTTOM), label_R.transpose(Image.FLIP_TOP_BOTTOM)
        return img_L, img_R, label_L, label_R


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_L, img_R, label_L, label_R):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(img_L, self.mean, self.std), F.normalize(img_R, self.mean, self.std), F.normalize(label_L, self.mean, self.std), F.normalize(label_R, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img_L, img_R, label_L, label_R):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img_L), F.to_tensor(img_R), F.to_tensor(label_L), F.to_tensor(label_R)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_L, img_R, label_L, label_R):
        for t in self.transforms:
            img_L, img_R, label_L, label_R = t(img_L, img_R, label_L, label_R)
            # label = t(label)
        return img_L, img_R, label_L, label_R

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
