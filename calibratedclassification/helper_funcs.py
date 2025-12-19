import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch

from PIL import Image
import re
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_STATS = (IMAGENET_MEAN, IMAGENET_STD)

class SquareCropAndResize(torch.nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size

        #print("W: %d H: %d" % (w, h))
        min_dim = min(h, w)
        img = torchvision.transforms.functional.center_crop(img, min_dim)

        if self.size is None:
            return img
        else:
            return torchvision.transforms.functional.resize(img, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

def get_inv_normalize( IMAGENET_NORMALIZATION_STATS = DEFAULT_STATS ):
    tfms = T.Normalize(   mean= [-m/s for m, s in zip(IMAGENET_NORMALIZATION_STATS[0], IMAGENET_NORMALIZATION_STATS[1])],
                       std= [1/s for s in IMAGENET_NORMALIZATION_STATS[1]])
    return tfms
    
denorm = get_inv_normalize()

data_transform = T.Compose([
    #squarecrop,
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Assuming images are in range [0, 1]
])

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.class_to_idx = {}
        self.classes = []

        pattern = re.compile(r'\.jpg$')
        for idx, cls_name in enumerate(sorted(os.listdir(root_dir))):
            if cls_name.startswith('.'):  # ✅ SKIP hidden directories
                continue
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue

            valid_images = [
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if pattern.search(fname)
            ]

            if valid_images:
                self.class_to_idx[cls_name] = len(self.classes)
                self.classes.append(cls_name)
                self.images.extend([(img_path, self.class_to_idx[cls_name]) for img_path in valid_images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
