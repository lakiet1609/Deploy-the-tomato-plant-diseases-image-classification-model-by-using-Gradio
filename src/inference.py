import torch
import os
import matplotlib.pyplot as plt

from typing import List, Tuple
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict
from timeit import default_timer as timer

from model.cnn_model import CNNModel
from utility.utils import find_classes
from config.cfg import TomatoConfig as tc

class Prediction:
    def __init__(self,
                 output_dir: str,
                 output_save_model: str,
                 image_size: Tuple[int, int],
                 transform: transforms = None):
        
        self.classes, _ = find_classes(tc.test_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.transform = transform
        self.model_path = os.path.join(output_dir, output_save_model)
        self.model = torch.load(self.model_path)


    def pred_and_plot_image(self, image_path):
        img = Image.open(image_path)

        if self.transform is not None:
            image_transform = self.transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

        self.model.to(self.device)
        self.model.eval()
        with torch.with_no_grad():
            transformed_image = image_transform(img).unsqueeze(dim=0)

            target_image_pred = self.model(transformed_image.to(self.device))

        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        return target_image_pred_probs, target_image_pred_label