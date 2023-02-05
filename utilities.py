# Libraries.
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as T
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from fvcore.nn import FlopCountAnalysis, flop_count_str
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Retrieves the CIFAR10 dataset.
def get_cifar10(train_transform = None, test_transform = None):

  # Downloading the training set.
  train_dataset =  datasets.CIFAR10(
    root = "CIFAR10",
    train = True,
    transform = train_transform,
    download = True
  )

  # Downloading the test set.
  test_dataset = datasets.CIFAR10(
    root = "CIFAR10",
    train = False,
    transform = test_transform,
    download = True
  )

  # Returning the two sets.
  return train_dataset, test_dataset

# Fuction that creates train, validation and test data loaders.
def create_data_loaders(train_transform, 
                        test_transform, 
                        img_size = 224, 
                        split = (0.8, 0.2), 
                        batch_size = 32, 
                        num_workers = 1):

  # Retrieving CIFAR10.
  train_dataset, test_dataset = get_cifar10(train_transform, test_transform)

  # Splitting train_dataset to create a validation set.
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                             (int(len(train_dataset) * split[0]), 
                                                              int(len(train_dataset) * split[1])))

  # Train data loader.
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    pin_memory = True,
    drop_last = True,
    sampler = None
  )

  # Validation data loader.
  val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    pin_memory = True,
    drop_last = False,
    sampler = None
  )

  # Test data loader.
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    pin_memory = True,
    drop_last = False,
    sampler = None
  )

  # Returning the three data loaders.
  return train_loader, val_loader, test_loader
  
# Computes and prints number of FLOPs.
def get_flops(model, shape):

  # Computing number of FLOPs.
  flops = FlopCountAnalysis(model, (torch.randn(shape),))

  # Printing number of FLOPs per layer.
  print(flop_count_str(flops))
  
# Creates the objects used for training the models.
def get_training_objects(model, 
                         epochs = 25, 
                         lr = 6e-5, 
                         wd = 1e-5, 
                         loss = nn.CrossEntropyLoss, 
                         opt = torch.optim.AdamW, 
                         sched = torch.optim.lr_scheduler.CosineAnnealingLR):
  
  # Epochs.
  EPOCHS = epochs

  # Initial learning rate.
  LR = lr

  # Weight decay.
  WD = wd

  # Loss function.
  loss_fn = loss()

  # Optimizer.
  optimizer = opt(model.parameters(), lr = LR, weight_decay = WD)

  # Scheduler.
  scheduler = sched(optimizer, EPOCHS)

  # Returning objects.
  return EPOCHS, loss_fn, optimizer, scheduler
  
# Trains the model.
def train(model, optimizer, scheduler, loss_fn, train_loader, val_loader, epochs, device = "cuda", history = None):

  # Metrics.
  if history == None: 
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

  # Iterating over epochs.
  for epoch in range(epochs):

    print("Epoch %2d/%2d:" % (epoch + 1, epochs))

    # Training.
    model.train()

    # Metrics
    train_loss = 0.0
    num_train_correct = 0
    num_train_examples = 0

    # Iterating over mini-batches.
    for batch in tqdm(train_loader, desc = "Training", position = 0):

      # Gradient reset.
      optimizer.zero_grad()

      # Moving x and y to GPU.
      x = batch[0].to(device)
      y = batch[1].to(device)

      # Predictions.
      yhat = model(x)

      # Loss.
      loss = loss_fn(yhat, y)

      # Computing the gradient.
      loss.backward()

      # Updating the parameters.
      optimizer.step()

      # Updating the metrics.
      train_loss += loss.item() * x.shape[0]
      num_train_correct += (torch.argmax(yhat, 1) == y).sum().item()
      num_train_examples += x.shape[0]

    # Computing the epoch's metrics.
    train_accuracy = num_train_correct / num_train_examples
    train_loss = train_loss / len(train_loader.dataset)

    # Saving learning rate.
    lr = scheduler.get_last_lr()[0]

    # Updating the learning rate.
    scheduler.step()

    # Validation.
    model.eval()

    # Metrics.
    val_loss = 0.0
    num_val_correct = 0
    num_val_examples = 0

    with torch.no_grad():

      # Iterating over mini-batches.
      for batch in tqdm(val_loader, desc = "Validation", position = 0):

        # Moving x and y to GPU.
        x = batch[0].to(device)
        y = batch[1].to(device)

        # Predictions.
        yhat = model(x)

        # Loss.
        loss = loss_fn(yhat, y)

        # Updating the metrics.
        val_loss += loss.item() * x.shape[0]
        num_val_correct += (torch.argmax(yhat, 1) == y).sum().item()
        num_val_examples += y.shape[0]

      # Computing the epoch's metrics.
      val_accuracy = num_val_correct / num_val_examples
      val_loss = val_loss / len(val_loader.dataset)

    print("{}train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}, lr: {:.4e}{}".format("\n" if epoch == epochs - 1 else "",
                                                                                                                      train_loss, 
                                                                                                                      train_accuracy, 
                                                                                                                      val_loss, 
                                                                                                                      val_accuracy,
                                                                                                                      lr,
                                                                                                                      "\n" if epoch != epochs - 1 else ""))

    # Appending the epoch's metrics to history lists.
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)
    history["lr"].append(lr)

  # Returning history.
  return history
  
# Loads a checkpoint.
def load_checkpoint(path, model, optimizer, scheduler):

  # Loading checkpoint.
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  scheduler.load_state_dict(checkpoint["scheduler"])

  # Returning checkpoint entries.
  return model, optimizer, scheduler
  
# Tests a model.
def test(model, test_loader, device = "cuda"):
  
  # Testing.
  model.eval()

  # Predictions and ground truth.
  y_pred = []
  y_true = []

  with torch.no_grad():

    # Iterating over mini-batches.
    for batch in tqdm(test_loader, desc = "Testing", position = 0):

      # Moving x and y to GPU.
      x = batch[0].to(device)
      y = batch[1].to(device)

      # Predictions.
      yhat = model(x)

      # Updating variables.
      y_pred.extend(torch.argmax(yhat, 1).tolist())
      y_true.extend(y.tolist())

  # Returning pred and true.
  return y_pred, y_true
  
# Reads files.
def read_output_files(m_names, resolution = "224x224"):

  # Training histories.
  histories = {}

  # Test accuracies.
  test_acc = {}

  # Reading histories and test accuracies.
  for name in m_names:
    test_acc[name] = float(np.load(f"{name}_accuracy_{resolution}.npy"))
    with open(f"{name}_history_{resolution}.pkl", "rb") as f: 
      histories[name] = pickle.load(f)
  
  # Returning histories and test_acc.
  return histories, test_acc

# Plots the validation losses and accuracies.
def plot_validation(m_names, histories, xy1, xy2):

  # Creating the figure and axes.
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 4))

  # Inset plots.
  ax1_ins = inset_axes(ax1, 0.75, 1, loc = "upper left", bbox_to_anchor = (0.4, 0.85), bbox_transform = ax1.transAxes)
  ax2_ins = inset_axes(ax2, 0.75, 1, loc = "upper left", bbox_to_anchor = (0.4, 0.55), bbox_transform = ax2.transAxes)

  # Iterating over models.
  for name in m_names:

    # Computing the x axis array.
    x = np.linspace(1, len(histories[name]["val_loss"]), len(histories[name]["val_loss"]), dtype = int)

    # Plotting.
    ax1.plot(x, histories[name]["val_loss"], label = r"${}$".format(name))
    ax2.plot(x, histories[name]["val_accuracy"], label = r"${}$".format(name))
    ax1_ins.plot(x, histories[name]["val_loss"])
    ax2_ins.plot(x, histories[name]["val_accuracy"])

  x1_min, x1_max, y1_min, y1_max = xy1
  x2_min, x2_max, y2_min, y2_max = xy2

  ax1_ins.set_xlim(x1_min, x1_max)
  ax1_ins.set_ylim(y1_min, y1_max)
  ax2_ins.set_xlim(x2_min, x2_max)
  ax2_ins.set_ylim(y2_min, y2_max)

  mark_inset(ax1, ax1_ins, loc1 = 1, loc2 = 3)
  mark_inset(ax2, ax2_ins, loc1 = 2, loc2 = 4)

  ax1.set_ylabel("Validation loss")
  ax1.set_xlabel("Epoch")
  ax1.set_xticks(x)
  ax1.legend()

  ax2.set_ylabel("Validation accuracy")
  ax2.set_xlabel("Epoch")
  ax2.set_xticks(x)
  ax2.legend()

  plt.show()
  
# Plots test accuracy vs. FLOPs.
def plot_accuracy_vs_flops(m_names, flops, test_acc):

  # Creating the figure and axes.
  fig, ax = plt.subplots(1, 1, figsize = (6.15, 4))

  # Scatter plot.
  ax.scatter(list(flops.values()), list(test_acc.values()), c = ["tab:blue", "tab:blue", "tab:blue", "tab:orange"])

  # Annotations.
  for model in m_names:
    ax.annotate(r"${}$".format(model), (flops[model], test_acc[model]))

  cmt_patch = mpatches.Patch(color = "tab:blue", label = "CMT-Ti")
  res_patch = mpatches.Patch(color = "tab:orange", label = "ResNet")

  ax.legend(handles = [cmt_patch, res_patch])

  ax.set_ylabel("Test accuracy")
  ax.set_xlabel("FLOPs")

  plt.show()