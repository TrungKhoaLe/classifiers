import torch
from torchvision import transforms


class ImageStem:
    def __init__(self):
        self.pil_transforms = transforms.Compose([])
        self.pil_to_tensor = transforms.ToTensor()
        self.torch_transforms = torch.nn.Sequential()

    def __call__(self, img):
        img = self.pil_transforms(img)
        img = self.pil_to_tensor(img)

        with torch.no_grad():
            img = self.torch_transforms(img)

        return img


class FlowerStem(ImageStem):
    def __init__(self):
        self.torch_transforms = torch.nn.Sequential(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
