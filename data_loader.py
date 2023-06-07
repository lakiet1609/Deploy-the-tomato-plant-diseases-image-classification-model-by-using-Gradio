import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from custom_dataset import custom_folder, data_augmentation

def load_data(train_dir: str,
              test_dir: str,
              batch_size: int,
              num_workers: int,
              image_size: int):
    
    train_transformation, test_transformation = data_augmentation(image_size=image_size)
    
    train_data = custom_folder(target_dir=train_dir,
                               transform=train_transformation)
    
    class_weights = []
    for root, subdir, files in os.walk(train_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    
    sample_weights = [0] * len(train_data)
    
    for idx, (data,label) in enumerate(train_data):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    test_data = custom_folder(target_dir=test_dir,
                              transform=test_transformation)
       
    #Turn into dataloaders
    load_train_data = DataLoader(train_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 sampler=sampler)
    
    load_test_data = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    
    return load_train_data, load_test_data


# if __name__ == '__main__':
#     load_data(train_dir=r'data\train', test_dir=r'data\test', batch_size=32, num_workers=6, image_size=224)