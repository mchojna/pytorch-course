"""
Makes prediction with a PyTorch image classification model
"""
import argparse

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",   
)

parser.add_argument("-m", "--model_path", default="models/05_pytorch_going_modular_script_mode.pth")
parser.add_argument("-i", "--image_path", required=True)

args = parser.parse_args()

MODEL_PATH = args.model_path
IMAGE_PATH = args.image_path

import utils
import model_builder
import torch
from torchvision import transforms
from PIL import Image

img = Image.open(IMAGE_PATH)

model = utils.load_model(
    input_shape=3, 
    output_shape=3, 
    hidden_units=8, 
    model_path=MODEL_PATH,
)

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

img_tensor = data_transform(img)

model.eval()

with torch.inference_mode():
    y_pred_logits = model(img_tensor.unsqueeze(dim=0))

y_pred_probs = torch.softmax(y_pred_logits, dim=1)
y_pred_label = torch.argmax(y_pred_probs, dim=1)

print(y_pred_label.item())
