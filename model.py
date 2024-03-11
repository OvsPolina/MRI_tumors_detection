import torch
import torch.nn as nn

import const

import matplotlib.pyplot as plt

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Encoder (VGG)
        self.encoder = VGG('VGG16')
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder.features(x)
        # Decoder
        x = self.decoder(x)
        #x = torch.sigmoid(x)
        #x = torch.round(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #print("Inputs = ", inputs.shape, ", Label = ", labels.shape)

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(inputs)  # Получаем предсказания модели
            #print("Outputs = ", outputs.shape)
            loss = criterion(outputs, labels)  # Вычисляем функцию потерь
            loss.backward()  # Распространяем градиенты
            optimizer.step()  # Обновляем веса

            train_loss += loss.item() * inputs.size(0)
            o, predicted = torch.max(outputs , 1)
            train_total += labels.size(0)
            train_correct += (predicted==labels).sum().item()


        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                o, predicted = torch.max(outputs , 1)
                val_total += labels.size(0)
                val_correct += (predicted==labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_loss_history.append(val_loss)
            val_accuracy = val_correct / val_total
            val_acc_history.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), const.model_out)

    history = {'train_loss': train_loss_history, 'val_loss': val_loss_history, 'train_acc': train_acc_history, 'val_acc': val_acc_history}
    plot_accuracy_and_loss(history)

    print('Training complete')


def plot_accuracy_and_loss(history):
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.show()

    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

def test_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    model.to(device)

    ious = []  # Список для хранения значений IoU
    total_iou = 0.0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
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

    #print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    test_accuracy = 0

    return test_loss, test_accuracy



def visualize_predictions(model, test_loader, device='cpu'):
    model.eval()
    total_images = 4
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(device))

    fig, axes = plt.subplots(nrows=total_images, ncols=3, figsize=(10, 20))

    for i in range(total_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original Image")

        pred_mask = torch.round(outputs[i]).squeeze().cpu().numpy()
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title("Predicted Mask")

        true_mask = labels[i].squeeze().cpu().numpy()
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title("True Mask")

    plt.tight_layout()
    plt.show()


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
