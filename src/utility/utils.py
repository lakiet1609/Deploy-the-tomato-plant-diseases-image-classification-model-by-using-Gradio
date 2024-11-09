import torch
import os

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#Plot 1 batch of image
def visualize_image(x_batch, 
                    y_batch, 
                    batch_size, 
                    class_idx):
    
    fig = plt.figure(figsize=(20,8))
    columns = 4
    rows = 4
    fig.subplots_adjust(wspace=1.2, hspace=0.3)
    for i in range(1, columns*rows):
        num = np.random.randint(batch_size)
        image = x_batch[num]
        fig.add_subplot(rows, columns, i)
        label = int(np.argmax(y_batch[num]))
        plt.title(f'{label}: {[k for k, v in class_idx.items() if v == label]}')
        plt.imshow(image.permute(1,2,0))
        plt.axis(False)
    plt.show()


#Evaluation curve
def plot_evaluation_curve(results):
    train_loss = torch.tensor(results['train_loss']).to('cpu')
    test_loss = torch.tensor(results['test_loss']).to('cpu')
    train_accuracy = torch.tensor(results["train_acc"]).to('cpu')
    test_accuracy = torch.tensor(results["test_acc"]).to('cpu')
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
    #Plot accuracy
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
#Save model
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def find_classes(target_dir):
    classes = sorted(sub_dir.name for sub_dir in os.scandir(target_dir) if sub_dir.is_dir())
    if not classes:
        raise FileNotFoundError(f'Could not find any classes in {target_dir}')
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

     
