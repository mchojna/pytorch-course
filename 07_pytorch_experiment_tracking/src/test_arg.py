"""
Trains a PyTorch image classification model using device-agnostic code and custom arguments
"""
import argparse

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("-r", "--train_dir", default="data/pizza_steak_sushi/train")
parser.add_argument("-s", "--test_dir", default="data/pizza_steak_sushi/test")
parser.add_argument("-l", "--learning_rate", default="0.001")
parser.add_argument("-b", "--batch_size", default="32")
parser.add_argument("-e", "--num_epochs", default="8")
parser.add_argument("-u", "--hidden_units", default="8")
parser.add_argument("-n", "--model_name", default="model.pth")

args = parser.parse_args()

TRAIN_DIR = args.train_dir
TEST_DIR = args.test_dir
LEARNING_RATE = float(args.learning_rate)
BATCH_SIZE = int(args.batch_size)
NUM_EPOCHS = int(args.num_epochs)
HIDDEN_UNITS = int(args.hidden_units)
MODEL_NAME = args.model_name

print(f"Paths:\n\tTrain directory: {TRAIN_DIR}\n\tTest directoryt: {TEST_DIR}\n\tModel directory: models/{MODEL_NAME}\n")
print(f"Hyperparameters:\n\tLearning rage: {LEARNING_RATE}\n\tBatch size: {BATCH_SIZE}\n\tNumber of epochs: {NUM_EPOCHS}\n\tHidden units: {HIDDEN_UNITS}\n")

import os
import torch
from torchvision import transforms
import setup_data, engine, model_builder, utils
from timeit import default_timer as timer

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.mps.is_available else "cpu"
)

if __name__ == "__main__":

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ])

    # Create DataLoaders and get class_names
    test_dataloader, train_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
    )

    # Start timer
    start_time = timer()

    # Start trainig with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )

    # End timer
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model to file
    utils.save_model(
        model=model,
        target_dir="models",
        model_name=MODEL_NAME
    )
