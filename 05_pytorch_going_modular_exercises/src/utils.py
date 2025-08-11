"""
File containint utility functions for model training
"""

from pathlib import Path
import torch

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
):
    """Saves a PyTorch model to a target directory.


    Args:
        model (torch.nn.Module): A target PyTorch model to save.
        target_dir (str): A directory for saving the model to.
        model_name (str): A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
