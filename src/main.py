import torch
import os
import torch.nn as nn

from timeit import default_timer as timer 
from torchmetrics import Accuracy, Recall

from model.cnn_model import CNNModel
from ingest.custom_dataset import find_classes
from ingest.custom_dataset import CustomDataLoader
from ingest.engine import Engine
from utility.utils import plot_evaluation_curve, save_model
from config.cfg import TomatoConfig as tc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Get class
    classes, class_to_idx = find_classes(tc.train_path)
    num_classes = len(classes)

    # Create dataloader
    custom_data_loader = CustomDataLoader(train_dir=tc.train_path,
                                          test_dir=tc.test_path,
                                          batch_size=tc.batch_size,
                                          image_size=tc.image_size)
    
    train_data, test_data = custom_data_loader.load_data()

    # Create model
    CNN_model = CNNModel(input_param=tc.channels, output_param=num_classes)
    
    #Compile model
    loss_function = nn.CrossEntropyLoss()
    accuracy = Recall(task="multiclass", average='micro', num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(params=CNN_model.parameters(), lr=tc.learning_rate)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=tc.optim_step_size)

    #Training progress
    start_time = timer()
    trainer = Engine(model=CNN_model,
                     train_data_loader=train_data,
                     test_data_loader=test_data,
                     loss_function=loss_function,
                     optimizer=optimizer,
                     epochs=tc.epochs,
                     accuracy=accuracy)
    
    model_training = trainer.training()
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    #Plot evaluation curve
    plot_evaluation_curve(model_training)

    #Save model
    save_model(model=CNN_model,
               target_dir=tc.output_dir,
               model_name=tc.output_save_model)


if __name__== '__main__':
    main()