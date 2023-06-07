import torch
import os
import cnn_model 
from custom_dataset import find_classes
from data_preprocess import custom_folder
from data_loader import load_data
from utils import visualize_image, plot_evaluation_curve, save_model
from engine import training
from prediction import pred_and_plot_image, predict_gradio
from torchmetrics import Accuracy, Recall
import torch.nn as nn
from timeit import default_timer as timer 
from PIL import Image

def main():
    #Split folder in to right form
    data_path = 'data'
    # custom_folder(data_path)
    
    #Set up directory
    train_path = r'data/train'
    test_path = r'data/test'

    #Set up parameters
    image_size = 128
    channels = 3
    epochs = 6
    batch_size = 32
    learning_rate = 0.001
    num_workers = os.cpu_count()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Get class
    classes, class_to_idx = find_classes(train_path)
    num_classes = len(classes)
    print(classes)

    # Create dataloader
    # load_train_data, load_test_data = load_data(train_dir=train_path,
    #                                               test_dir=test_path,
    #                                               batch_size=batch_size,
    #                                               image_size=image_size,
    #                                               num_workers=num_workers)

    # #Get 1 batch_size in training data
    # x_batch, y_batch = next(iter(load_train_data))
     
    # #Visualize 1 batch of image
    # visualize_image(x_batch=x_batch,
    #                 y_batch=y_batch,
    #                 batch_size=batch_size,
    #                 class_idx=class_to_idx)
    
    #Create model
    # model_1 = cnn_model.custom_model(input_param=channels, output_param=num_classes)
    
    # #Compile model
    # loss_function = nn.CrossEntropyLoss()
    # accuracy = Recall(task="multiclass", average='micro', num_classes=num_classes).to(device)
    # optimizer = torch.optim.Adam(params=model_1.parameters(), lr=learning_rate)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    # #Training progress
    # start_time = timer()
    # model_1_results = training(model=model_1, 
    #                            train_dataloader=load_train_data,
    #                            test_dataloader=load_test_data,
    #                            optimizer=optimizer,
    #                            loss_function=loss_function,
    #                            accuracy=accuracy, 
    #                            epochs=epochs,
    #                            device=device)
    # end_time = timer()
    # print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    # #Plot evaluation curve
    # plot_evaluation_curve(model_1_results)

    # #Save model
    # save_model(model=model_1,
    #            target_dir='models',
    #            model_name='model_2.pth')
    
    #Load model
    # model_path = r'models\model_2.pth'
    # loaded_model_2 = cnn_model.custom_model(input_param=channels, output_param=num_classes) 

    # # Load in the saved state_dict()
    # loaded_model_2.load_state_dict(torch.load(f=model_path))

    # # Send model to GPU
    # loaded_model_2 = loaded_model_2.to(device)
    
    # # Make predictions on and plot the images
    # image_path = r'data\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\0c546c5e-f03f-4012-91cb-f036b2dae385___UF.GRC_YLCV_Lab 01783.JPG'
    # pred_and_plot_image(model= loaded_model_2, 
    #                     image_path= image_path, 
    #                     class_names=classes,
    #                     image_size=image_size,
    #                     device= device)
    
    # Predict on the target image and print out the outputs
    # image_path = r'data\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\0c546c5e-f03f-4012-91cb-f036b2dae385___UF.GRC_YLCV_Lab 01783.JPG'
    # img_obj = Image.open(image_path)
    # pred_prob, pred_time = predict_gradio(img_obj)
    # print(pred_prob)
    # print(pred_time)


if __name__== '__main__':
    main()