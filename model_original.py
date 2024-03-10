import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn

import matplotlib.pyplot as plt
import numpy as np

from tqdm.autonotebook import tqdm

#Define our helper function that creates a hidden layer for a CNN
def cnnLayer(in_filters, out_filters, kernel_size=3):
    """
    in_filters: how many channels are in the input to this layer
    out_filters: how many channels should this layer output
    kernel_size: how large should the filters of this layer be
    """
    padding = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.BatchNorm2d(out_filters),
        nn.LeakyReLU(), # I'm not setting the leak value to anything just to make the code shorter.
    )

class EarlyStopping:
  def __init__(self, patience:int=10, delta:float=0.001, path='best_model.pth'):
    self.patience = patience
    self.delta = delta
    self.path = path
    self.best_score = None
    self.early_stop = False
    self.counter = 0

  def __call__(self,val_loss , model):
    if self.best_score is None:
      self.best_score = val_loss
      self.save_checkpoint(model)

    elif val_loss > self.best_score + self.delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True

    else:
      self.best_score = val_loss
      self.save_checkpoint(model)
      self.counter = 0

  def save_checkpoint(self,model):
    torch.save(model.state_dict(),self.path)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_scheduler,
               device = 'cpu'):
    model.train()
    train_loss = 0
    train_accuracy = 0
    print("train_step")

    for batch, (X,y) in enumerate(tqdm(dataloader, desc='Training')):
        X = X.to(device,dtype=torch.float32)
        y = y.to(device,dtype=torch.float32) # dtype of the target y to torch.long assuming it represents class indices. This is the appropriate datatype for class labels.

        optimizer.zero_grad()

        y_pred_logits = model(X)

        loss= loss_fn(y_pred_logits,y)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        #train_accuracy += balanced_accuracy_score(y.cpu().numpy(),y_pred_target.detach().cpu().numpy(),adjusted=True)

    lr_scheduler.step()

    train_loss = train_loss / len(dataloader)
    #train_accuracy = train_accuracy / len(dataloader)

    return train_loss#, train_accuracy

def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device = 'cpu'):

    model.eval()
    val_loss = 0
    val_accuracy = 0
    print("val_step")

    with torch.no_grad():
        for batch , (X,y) in enumerate (tqdm(dataloader, desc='Validation')):
            X = X.to(device,dtype=torch.float32)
            y = y.to(device,dtype=torch.float32)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits,y)
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    #val_accuracy = val_accuracy / len(dataloader)

    return val_loss #, val_accuracy

def Train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          early_stopping,
          lr_scheduler=None,  # Добавлено значение по умолчанию для lr_scheduler
          epochs: int=10,
          device='cpu'):
    results = {
        'train_loss' : [],
        'train_accuracy' : [],
        'val_loss' : [],
        'val_accuracy' : []}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,  # Исправлено на правильное использование
                                device=device)

        val_loss = val_step(model=model,
                                          dataloader=val_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        #print(f'Validation Accuracy: {val_accuracy:.4f}')
        early_stopping(val_loss,model)

        if early_stopping.early_stop == True:
            print('Early stopping')
            break

        results['train_loss'].append(train_loss)
        #results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        #results['val_accuracy'].append(val_accuracy)

    return results

def manual_seed(seed):
  torch.cuda.manual_seed(seed)
  torch.manual_seed(seed)



def count_correct_predictions(outputs, labels, threshold=0.8):
    correct = 0

    for output, label in zip(outputs, labels):
        # Преобразование выхода модели и метки в формат [0, 1]
        output = torch.sigmoid(output)
        label = label.float()

        # Подсчет пикселей, совпадающих с маской
        mask_pixels = torch.sum(label)
        correct_pixels = torch.sum(torch.round(output) == label)
        #print(mask_pixels, correct_pixels)

        # Проверка условия для правильно помеченной маски
        if correct_pixels >= threshold * mask_pixels:
            correct += 1

    return correct


def calculate_iou(prediction, target):
    intersection = torch.logical_and(prediction, target).sum()
    union = torch.logical_or(prediction, target).sum()
    iou = intersection / union
    return iou

def test(model, test_loader, loss_fn, device='cpu'):
    model.eval()
    test_loss = 0.0
    total_iou = 0.0
    total_samples = len(test_loader.dataset)
    losses = []  # Список для хранения потерь на каждом батче
    ious = []  # Список для хранения значений IoU

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            losses.append(loss.item())  # Добавляем текущее значение потерь в список

            # Вычисляем IoU для каждой пары масок
            batch_iou = 0.0
            for pred_mask, true_mask in zip(outputs, labels):
                iou = calculate_iou(pred_mask.cpu(), true_mask.cpu())
                #print(iou)
                batch_iou += iou
                ious.append(iou)
            total_iou += batch_iou / len(outputs)

    accuracy = total_iou / total_samples * 100
    test_loss /= len(test_loader)
    print('Test Loss: {:.4f}, IoU Accuracy: {:.2f}%'.format(test_loss, accuracy))

    # Построение графика
    plt.plot(losses, label="Test Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Test Loss Over Batches")
    plt.legend()
    plt.show()

    # Построение гистограммы значений IoU
    plt.hist(ious, bins=20, range=(0, 1))
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.title("IoU Histogram")
    plt.show()

    # Вывод статистики по IoU
    print('Mean IoU:', np.mean(ious))
    print('Std IoU:', np.std(ious))


def visualize_predictions(model, test_loader, device='cpu'):
    model.eval()
    total_images = 4
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(device))

    fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(10, 20))

    for i in range(total_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original Image")
        pred_mask = torch.round(outputs[i]).squeeze().cpu().numpy()
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()


