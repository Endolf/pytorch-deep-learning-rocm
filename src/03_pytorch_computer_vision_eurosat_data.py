import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets
from torchvision import transforms

import random

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import requests
from pathlib import Path

from timeit import default_timer as timer

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch torchvision version: {torchvision.__version__}")

dataset = datasets.EuroSAT(
    root="data",
    download=True,
    transform = torchvision.transforms.ToTensor(),
    target_transform = None
)

train_data, test_data = random_split(dataset=dataset, lengths=[0.8,0.2], generator=torch.Generator().manual_seed(42))

print(f"Training data length: {len(train_data)}, Test data length: {len(test_data)}")

torch.manual_seed(42)
random.seed(42)

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

def plot_random_sample(rows: int, cols: int, data: list):
    test_samples = []
    test_labels = []
    to_pil_image_transformer = transforms.ToPILImage()

    for sample, label in random.sample(data, k=cols*rows):
        test_samples.append(sample)
        test_labels.append(label)

    fig = plt.figure(figsize=(10,10))
    for i, sample in enumerate(test_samples):
        image, label_index = train_data[i]
        label = dataset.classes[label_index]
        plt.subplot(rows, cols, i+1)
        plt.imshow(to_pil_image_transformer(image))
        plt.title(label, fontsize=8)
        plt.axis(False)
    fig.tight_layout()
    plt.show()

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
print(f"Torch device: {device}")
if torch.cuda.is_available():
    for device_id in range(torch.cuda.device_count()):
        print(f"Found CUDA device: cuda:{device_id} - {torch.cuda.get_device_name(device_id)}")

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import accuracy_fn

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model prediction on data_loader"""
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Accumulate loss and acc per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc to find average per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc
    } # only works when the model was created with a class

def make_predictions(model: nn.Module,
                     data: list,
                     device: torch.device = device):

    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in tqdm(data, "Making predictions..."):
            # Add a batch dimension and pass to target device
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    # Return the prediction probablities as a tensor
    return torch.stack(pred_probs)

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints differene between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimiser: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training step with model trying to learn on data_loader"""

    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%")

def nn_test_step(model: torch.nn.Module,
                 device: torch.device,
                 data_loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 accuracy_fn,
                 ):
    """Performs a testing step with model trying to test the trained model using data_loader"""
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}%")

class EuroSATModel_0(nn.Module):
    """
    Model architecture that replicates the TinyVGG model from CNN explainer
    website.
    """
    def __init__(self, input_shape:int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            # nn.Flatten(start_dim=0),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, #Output of conv_block_2 assuming input image is 28x28
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        # print(f"Output shape of conv_block_1 {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of conv_block_2 {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier {x.shape}")
        return x

plot_random_sample(5,5,list(test_data))

torch.manual_seed(42)
model_0 = EuroSATModel_0(input_shape=3,
                         hidden_units=10,
                         output_shape=len(dataset.classes)).to(device)

# Setup loss function
loss_fn = nn.CrossEntropyLoss()

# Setup optimiser function
optimiser = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

train_time_start = timer()

EPOCHS = 30

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}")

    train_step(model=model_0, data_loader=train_dataloader, loss_fn=loss_fn,
               optimiser=optimiser, device=device,
               accuracy_fn=accuracy_fn)

    nn_test_step(model=model_0, device=device, data_loader=test_dataloader,
                 loss_fn=loss_fn, accuracy_fn=accuracy_fn)

train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start,
                                            end=train_time_end,
                                            device=str(next(model_0.parameters()).device))

model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn)

print(f"Model 0 evaluation: {model_0_results}")

test_samples = []
test_labels = []

cols, rows = 5, 5
for sample, label in random.sample(list(test_data), k=cols*rows):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=model_0, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

to_pil_image_transformer = transforms.ToPILImage()
fig = plt.figure(figsize=(10,10))
for i, sample in enumerate(test_samples):
    plt.subplot(rows, cols, i+1)
    plt.imshow(to_pil_image_transformer(sample.squeeze()))
    predicted_label = dataset.classes[pred_classes[i]]
    truth_label = dataset.classes[test_labels[i]]
    # Set the title to *predicted label (actual label)*
    if predicted_label == truth_label:
        colour = "g"
        title_text = f"{predicted_label}"
    else:
        colour = "r"
        title_text = f"{predicted_label}\n({truth_label})"
    plt.title(title_text, fontsize="9", c=colour)
    plt.axis(False)
fig.tight_layout()
fig.show()

y_preds = []
targets = []
model_0.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        X, y = X.to(device), y.to(device)
        y_logits = model_0(X)
        #Turn logits to preds to labels
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())
        targets.append(y)

y_pred_tensor = torch.cat(y_preds)
targets_tensor = torch.cat(targets).to("cpu")

conf_mat = ConfusionMatrix(num_classes=len(dataset.classes), task="multiclass")
conf_mat_tensor = conf_mat(preds=y_pred_tensor, target=targets_tensor) #test_data.targets are the numerical labels in the test_data

fig, axis = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(),
                                  class_names=dataset.classes,
                                  figsize=(10,7)
                                  )
fig.show()