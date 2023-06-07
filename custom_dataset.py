from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms

def find_classes(target_dir):
    classes = sorted(sub_dir.name for sub_dir in os.scandir(target_dir) if sub_dir.is_dir())
    if not classes:
        raise FileNotFoundError(f'Could not find any classes in {target_dir}')
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def data_augmentation(image_size: int):
    train_transformation = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transformation = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return train_transformation, test_transformation

class custom_folder(Dataset):
    def __init__(self, target_dir, transform= None):
        format_type = ['*.JPG','*.jpg']
        for i in format_type:
            self.paths = list(Path(target_dir).glob(f'*/{i}'))
            self.transform = transform
            self.classes, self.class_to_idx = find_classes(target_dir) 
    
    def load_image(self, idx: int):
        image_path = self.paths[idx]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = self.load_image(idx=idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

