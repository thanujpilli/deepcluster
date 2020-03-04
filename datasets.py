import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, extensions=None):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if has_file_allowed_extension(path, extensions):
                item = (path, None)
                images.append(item)
    return images


class UnsupervisedDataset(Dataset):
    def __init__(self, root, transform=None):
        # super().__init__()
        self.root = root
        self.imgs = make_dataset(root, IMG_EXTENSIONS)
        self.transform = transform

    def __getitem__(self, idx):
        img_loc = self.imgs[idx][0]
        with open(img_loc, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        """
        unsupervised learning needs only the img but most part of the code base is reused from
        supervised form of training the model which requires a target . So we pass a dummy target(1) and will be ignored
        while unpacking as "img, _"
        """
        return img, 1

    def __len__(self):
        return len(self.imgs)
