from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from timeit import default_timer as timer
from cnn_model import custom_model
from custom_dataset import find_classes

classes, class_to_idx = find_classes(r'data\test')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (128, 128),
                        transform: transforms = None,
                        device: torch.device=device):
    
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    print(target_image_pred_probs)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    print(target_image_pred_label)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()
    

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
    model_path = r'models\model_2.pth'
    loaded_model_2 = custom_model(input_param=3, output_param=10) 
    loaded_model_2.load_state_dict(torch.load(f=model_path))
    loaded_model_2 = loaded_model_2.to(device)
    loaded_model_2.eval()
    
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = loaded_model_2(transformed_image.to(device))
    
    pred_probs = torch.softmax(target_image_pred, dim=1)
    pred_labels_and_probs = {classes[i]: float(pred_probs[0][i]) for i in range(len(classes))}
    pred_time = round(timer() - start_time, 4)

    return pred_labels_and_probs, pred_time