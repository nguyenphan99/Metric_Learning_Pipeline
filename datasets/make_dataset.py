
from torchvision import transforms
from torch.utils.data import Dataset
from .dataset import MVB
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(phase, input_size):
    if phase == "train":
        return  A.Compose([
                    A.LongestMaxSize(max_size = input_size ),
                    A.PadIfNeeded(min_height=input_size, min_width=input_size, p=1,value=0),
                    A.RandomResizedCrop(height= input_size, width=input_size,scale=(0.8,1.33)),
                    A.HorizontalFlip(p=0.5),
#                     A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
#                     A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
                    A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#                     A.CoarseDropout(p=0.5),
                    A.Normalize(),
                    ToTensorV2()
                ])
    elif phase in ['test','valid']:
        return A.Compose([
            A.LongestMaxSize(max_size = input_size),
            A.PadIfNeeded(min_height=input_size, min_width=input_size, p=1,value=0),
            A.Normalize(),
            ToTensorV2()
        ])
def make_dataset(traindf):
    dataloader =  MVB(traindf,transform = get_augmentation)
    return dataloader