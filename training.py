import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def _resume_skip(outdir, fname, save_model):
  """Resume helper: if a model checkpoint already exists, skip retraining
  (resumable across crashes). Returns the existing losses df if present, else an empty df.
  Disable with env SKIP_IF_EXISTS=0. Only skips when save_model is True."""
  if not save_model or os.environ.get('SKIP_IF_EXISTS', '1') != '1':
    return None, False
  model_path = f'{outdir}/{fname}_model.pt'
  if os.path.exists(model_path):
    print(f'[skip-exists] {model_path}')
    losses_path = f'{outdir}/{fname}_losses.csv'
    df = pd.read_csv(losses_path) if os.path.exists(losses_path) else pd.DataFrame()
    return df, True
  return None, False


def train_model(model, bag_loader, optimizer=None, scheduler=None, num_iter=1000, lr=0.0005, transform=None,
                outdir='/ewsc/hschluet/models/pbmc5/revision_rerun/', bar=True, device='cuda:7', fname='temp', plot=True,
                save_model=True, seed=12341, use_model_objective=True):
  _df, _skip = _resume_skip(outdir, fname, save_model)
  if _skip:
    return _df
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
                outdir='/ewsc/hschluet/models/pbmc5/revision_rerun/', use_model_objective=True,
                start=0, device='cuda:7', fname='temp', plot=True, save_model=True):
  _df, _skip = _resume_skip(outdir, fname, save_model)
  if _skip:
    return _df
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


def train_batched(model, bag_loader, num_steps, batch_bags=16, lr=1e-4, transform=None,
                  device='cuda:0', seed=12341):
    """Mini-batch training for a single aggregator (CE only, no router/load-balancing).
    Accumulates the loss over `batch_bags` bags per step, then one backward/step."""
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = model.to(device); model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    it = iter(bag_loader)
    hist = {'loss': [], 'err': []}
    for step in range(num_steps):
        opt.zero_grad()
        ce = 0.0; correct = 0
        for _ in range(batch_bags):
            bag, label = next(it)
            bag = bag.to(device); label = label.to(device)
            if transform is not None:
                bag = transform(bag)
            loss_b, yhat = model.calculate_objective(bag, label)
            ce = ce + loss_b
            correct += int((yhat == label).item())
        ce = ce / batch_bags
        ce.backward(); opt.step()
        hist['loss'].append(ce.item()); hist['err'].append(1 - correct / batch_bags)
    return hist


def train_moa(model, bag_loader, num_steps, batch_bags=16, lr=1e-4, lb_coef=0.01,
              warmup_frac=0.1, temp_start=2.0, temp_end=0.5, transform=None,
              device='cuda:0', seed=12341, bar=False):
    """Mixture-of-Aggregators training: CE + load-balancing loss over a mini-batch of
    `batch_bags` bags, Gumbel noise on router logits, staged dense->top-2 routing and
    temperature annealing over training."""
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = model.to(device); model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    it = iter(bag_loader)
    hist = {'loss': [], 'ce': [], 'lb': [], 'err': []}
    for step in range(num_steps):
        frac = step / num_steps
        temp = temp_start + (temp_end - temp_start) * frac
        dense = frac < warmup_frac
        opt.zero_grad()
        ce = 0.0; probs_list = []; correct = 0
        for _ in range(batch_bags):
            bag, label = next(it)
            bag = bag.to(device); label = label.to(device)
            if transform is not None:
                bag = transform(bag)
            logit, yhat, probs = model(bag, softmax=False, temperature=temp, gumbel=True, dense=dense)
            ce = ce + F.cross_entropy(logit.unsqueeze(0), label.unsqueeze(0))
            probs_list.append(probs)
            correct += int((yhat == label).item())
        ce = ce / batch_bags
        P = torch.stack(probs_list)                                     # [B, n_experts]
        topi = P.topk(model.top_k, dim=1).indices
        f = torch.zeros_like(P).scatter(1, topi, 1.0).mean(0)           # hard usage f_i
        lb = model.n_experts * (f * P.mean(0)).sum()                    # load-balancing loss
        loss = ce + lb_coef * lb
        loss.backward(); opt.step()
        hist['loss'].append(loss.item()); hist['ce'].append(ce.item())
        hist['lb'].append(lb.item()); hist['err'].append(1 - correct / batch_bags)
    return hist
