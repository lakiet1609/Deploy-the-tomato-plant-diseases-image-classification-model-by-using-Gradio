import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_param: int, output_param: int):
        super().__init__()      
        self.layer_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_param, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2)    
        )
        
        self.layer_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2)    
        )
        
        self.layer_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2)    
        )
        
        self.layer_block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*12*12, out_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=output_param)
        )
        
    def forward(self, x):
        x = self.layer_block1(x)
        x = self.layer_block2(x)
        x = self.layer_block3(x)
        x = self.layer_block4(x)
        return x


