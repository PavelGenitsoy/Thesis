import torch

from typing import Tuple
from tqdm import tqdm


def train(epoch, n_epochs, model, dl, loss_func, dev, optimizer, ds: int) -> Tuple[float, float]:
    model.train(True)
    torch.set_grad_enabled(True)

    epoch_loss, epoch_acc = 0, 0

    tq_batch = tqdm(dl, total=len(dl))
    for images, labels in tq_batch:
        images = images.to(dev)
        labels = labels.to(dev)

        optimizer.zero_grad()
        outs = model(images)
        _, preds = torch.max(outs, dim=1)

        loss = loss_func(outs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

        tq_batch.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        tq_batch.set_postfix_str(f'loss = {loss.item():.4f}')

    epoch_loss = epoch_loss / len(dl)
    epoch_acc = epoch_acc / ds

    return epoch_loss, epoch_acc


def evaluate(model, dl, loss_func, dev, ds: int) -> Tuple[float, float]:
    model.train(False)

    epoch_loss, epoch_acc = 0, 0

    for images, labels in tqdm(dl, total=len(dl), desc='Evaluate Model', leave=False):
        images = images.to(dev)
        labels = labels.to(dev)

        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_func(outputs, labels)

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss = epoch_loss / len(dl)
    epoch_acc = epoch_acc / ds

    return epoch_loss, epoch_acc
