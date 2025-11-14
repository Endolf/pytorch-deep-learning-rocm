"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
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

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    plt.show()

def prep_and_plot_confusion_matrix(model: nn.Module, data: Dataset, data_loader: DataLoader, device: torch.device):
    y_preds = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Making predictions..."):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            #Turn logits to preds to labels
            y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
            y_preds.append(y_pred.cpu())
            targets.append(y)

    y_pred_tensor = torch.cat(y_preds)
    targets_tensor = torch.cat(targets).to("cpu")

    conf_mat = ConfusionMatrix(num_classes=len(data.classes), task="multiclass")
    conf_mat_tensor = conf_mat(preds=y_pred_tensor, target=targets_tensor)

    fig, axis = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(),
                                      class_names=data.classes,
                                      figsize=(10,7)
                                      )
    fig.show()