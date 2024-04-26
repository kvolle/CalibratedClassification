import torch.nn.functional as F
import torchvision
import torch

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
