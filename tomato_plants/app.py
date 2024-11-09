import gradio as gr
import os
import torch
from src.model.cnn_model import custom_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from torchvision import transforms

class_name = ['Tomato___Bacterial_spot', 
              'Tomato___Early_blight', 
              'Tomato___Late_blight', 
              'Tomato___Leaf_Mold', 
              'Tomato___Septoria_leaf_spot', 
              'Tomato___Spider_mites Two-spotted_spider_mite', 
              'Tomato___Target_Spot', 
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
              'Tomato___Tomato_mosaic_virus', 
              'Tomato___healthy']

#Function for gradio
def predict_gradio(img):
    start_time = timer()
    
    image_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    
    #Load model
    model_path = 'model_2.pth'
    loaded_model_2 = custom_model(input_param=3, output_param=10) 
    loaded_model_2.load_state_dict(torch.load(f=model_path, map_location=torch.device("cpu")))
    loaded_model_2.eval()
    
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = loaded_model_2(transformed_image)
    
    pred_probs = torch.softmax(target_image_pred, dim=1)
    pred_labels_and_probs = {class_name[i]: float(pred_probs[0][i]) for i in range(len(class_name))}
    pred_time = round(timer() - start_time, 4)

    return pred_labels_and_probs, pred_time


#Create title
title = 'Tomato Plants Disease Detector'
description = 'A custom CNN image classification model to detect 9 diseases on tomato plants'
articile = 'Created at [Deploy the tomato plant diseases image classification by using Gradio](https://github.com/lakiet1609/Deploy-the-tomato-plant-diseases-image-classification-by-using-Gradio)'
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Create the Gradio demo
demo = gr.Interface(fn=predict_gradio,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=3, label='prediction'),
                             gr.Number(label='Prediction time (second)')],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=articile)

demo.launch(debug=False)
