"""
File containint utility functions for model training
"""

from pathlib import Path
import torch
import model_builder

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


def load_model(
    input_shape: int,
    hidden_units: int,
    output_shape: int,
    model_path: str,
) -> torch.nn.Module:
    """Loades a PyTorch model from a target directory.

    Args:
        input_shape (int): Input shape.
        hidden_units (int): Number of hidden units.
        output_shape (_type_): Output shape.

    Returns:
        torch.nn.Module: PyTorch module.
    """

    model = model_builder.TinyVGG(
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model
