"""
Contains functionality for creating PyTorch DataLoader's for 
image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0
BATCH_SIZE = 32


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders

    Takes in a training directory and testing directory path and turns them 
    into PyTorch Datasets and then into PyTorch DataLoaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision.transforms to perform on training and testing 
            data
        batch_size: Number of samples per batch in each of the DataLoaders
        num_workeres: Integer for number of workers per DataLoader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names) where 
        class_names is a list of the target classes

    Example usage:
        ```
        train_dataloader, test_dataloader, class_names = create_dataloader(
            train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            trainsform=some_transform,
            batch_size=32,
            num_workers=4
        )
        ```
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
        target_transform=None,
    )
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
        target_transform=None,
    )

    # Get class names as a list
    class_names = train_data.classes

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dataloader, test_dataloader, class_names
