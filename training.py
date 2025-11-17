import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def train_model(model, bag_loader, optimizer=None, scheduler=None, num_iter=1000, lr=0.0005, transform=None,
                outdir='/ewsc/hschluet/models/pbmc5/rerun/', bar=True, device='cuda:7', fname='temp', plot=True, 
                save_model=True, seed=12341, use_model_objective=True): 
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  train_loss = np.zeros(num_iter)
  train_error = np.zeros(num_iter)
  model = model.to(device)
  model.train()

  if optimizer is None:
    optimizer = optim.Adam(model.parameters(), lr=lr)

  tqdm_range = tqdm(np.arange(0, num_iter)) if bar else np.arange(0, num_iter)
  for i, (bag, label) in zip(tqdm_range, bag_loader):
    model.train()
    bag = bag.to(device=device)
    if transform is not None:
      bag = transform(bag)

    optimizer.zero_grad()
    if use_model_objective:
      loss, y_hat = model.calculate_objective(bag, label)
    else:  # for resnet
      y_logit = model(bag)
      loss = torch.nn.functional.cross_entropy(y_logit, label)
      y_hat = torch.argmax(y_logit, dim=-1)
    train_loss[i] = loss.item()  
    train_error[i] = 1.0 - (label == y_hat).float().mean().item()

    loss.backward()
    optimizer.step()

    if scheduler is not None:
      scheduler.step()

    if bar:
      tqdm_range.set_description(
        (
            f"iter: {i}; train loss: {train_loss[i]:.8f}; train error: {train_error[i]:.8f}"
        )
      )

  if save_model:
    torch.save(model.state_dict(), f'{outdir}/{fname}_model.pt')
    print(f'{outdir}/{fname}_model.pt')

  if plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(train_loss, label='training loss')
    ax2.plot(train_error, label='training error')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.show()

  df = pd.DataFrame()
  df['train loss'] = train_loss
  df['train error'] = train_error
  df.index.name = 'epoch'
  df.to_csv(f'{outdir}/{fname}_losses.csv')

  return df


def train_model_with_datasets(
                model, train_dataset, val_dataset=None, optimizer=None, batch_size=256, scheduler=None, num_epochs=30, lr=0.001,
                outdir='/ewsc/hschluet/models/pbmc5/rerun/', use_model_objective=True,
                start=0, device='cuda:7', fname='temp', plot=True, save_model=True):
  train_loss = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)
  train_acc = np.zeros(num_epochs)
  val_acc = np.zeros(num_epochs)
  model = model.to(device)
  model.train()

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  if val_dataset is not None:
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

  if optimizer is None:
    optimizer = optim.Adam(model.parameters(), lr=lr)

  tqdm_range = tqdm(np.arange(start, num_epochs))
  for e in tqdm_range:
    model.train()
    iter_loss = np.zeros(len(train_loader))
    iter_correct = 0
    for i, (imgs, labs) in enumerate(train_loader):
      optimizer.zero_grad()
      imgs = imgs.to(device)
      labs = labs.to(device)
      if use_model_objective:
        loss, y_hat = model.calculate_objective(imgs, labs)
      else:  # for resnet
        y_logit = model(imgs)
        loss = torch.nn.functional.cross_entropy(y_logit, labs)
        y_hat = torch.argmax(y_logit, dim=-1)
      loss.backward()
      iter_loss[i] = loss.item()
      iter_correct += (y_hat == labs).sum().item()
      optimizer.step()

    if scheduler is not None:
        scheduler.step()

    train_loss[e] = np.mean(iter_loss)
    train_acc[e] = iter_correct / len(train_dataset)

    if val_dataset is not None:
      model.eval()
      iter_loss = np.zeros(len(val_loader))
      iter_correct = 0
      for i, (imgs, labs) in enumerate(val_loader):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        labs = labs.to(device)
        with torch.no_grad():
          if use_model_objective:
            loss, y_hat = model.calculate_objective(imgs, labs)
          else:  # for resnet
            y_logit = model(imgs)
            loss = torch.nn.functional.cross_entropy(y_logit, labs)
            y_hat = torch.argmax(y_logit, dim=-1)
        iter_loss[i] = loss.item()
        iter_correct += (y_hat == labs).sum().item()

      val_loss[e] = np.mean(iter_loss)
      val_acc[e] = iter_correct / len(val_dataset)

    tqdm_range.set_description(
      (
          f"epoch: {e}; train loss: {train_loss[e]:.8f}; val loss: {val_loss[e]:.8f}; train accuracy: {train_acc[e]:.8f}; val accuracy: {val_acc[e]:.8f} "
      )
    )


  if save_model:
    torch.save(model.state_dict(), f'{outdir}/{fname}_model.pt')

  if plot:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_loss, label='train loss')
    ax.plot(val_loss, label='val loss')
    ax.plot(train_acc, label='train accuracy')
    ax.plot(val_acc, label='val accuracy')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.show()

  df = pd.DataFrame()
  df['train loss'] = train_loss
  df['val loss'] = val_loss
  df['train acc'] = train_acc
  df['val acc'] = val_acc
  df.index.name = 'epoch'
  df.to_csv(f'{outdir}/{fname}_losses.csv')

  return df
