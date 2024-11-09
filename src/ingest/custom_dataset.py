import os

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


from utility.utils import find_classes


class CustomDataset(Dataset):
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


class CustomDataLoader:
    def __init__(self,
                 train_dir: str,
                 test_dir: str,
                 batch_size: int,
                 image_size: int):
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.image_size = image_size


    def data_augmentation(self):
        train_transformation = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        test_transformation = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        return train_transformation, test_transformation


    def load_data(self):
        
        train_transformation, test_transformation = self.data_augmentation(image_size=self.image_size)
        
        train_data = CustomDataset(target_dir=self.train_dir,
                                   transform=train_transformation)
        
        class_weights = []
        for root, subdir, files in os.walk(self.train_dir):
            if len(files) > 0:
                class_weights.append(1/len(files))
        
        sample_weights = [0] * len(train_data)
        
        for idx, (data,label) in enumerate(train_data):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight
        
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        test_data = CustomDataset(target_dir=self.test_dir,
                                transform=test_transformation)
        
        #Turn into dataloaders
        load_train_data = DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    sampler=sampler)
        
        load_test_data = DataLoader(test_data,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)
        
        return load_train_data, load_test_data
