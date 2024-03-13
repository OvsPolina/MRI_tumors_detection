import dataset
import const
import model_original
import model as model_Unet

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.Grayscale(num_output_channels=3),  # Преобразовать в градации серого с тремя каналами
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Предполагая серое изображение
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

###############################  BRAIN DATASET ##########################################

# Create datasets
train_dataset = dataset.CustomDataset_brain(const.brain_path_train, transform=image_transform)
valid_dataset = dataset.CustomDataset_brain(const.brain_path_valid, transform=image_transform)
test_dataset = dataset.CustomDataset_brain(const.brain_path_test, transform=image_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Plot the images from the brain dataset
plt.figure(figsize=(10, 5))
plt.suptitle('Brain Dataset')

for i, (x, y) in enumerate(train_loader):
    for j in range(4):
        plt.subplot(2, 4, i * 4 + j + 1)
        plt.imshow(x[j][0], cmap='gray')
        plt.title(f'Image {i * 4 + j + 1}')
        plt.axis('off')
    if i == 1:  # Display only first batch
        break

plt.tight_layout()
plt.show()

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.Grayscale(num_output_channels=3),  # Преобразовать в градации серого с тремя каналами
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])


###############################  LIVER DATASET ##########################################

# Create datasets
train_dataset = dataset.CustomDataset_liver(const.liver_path_train, transform=image_transform)
valid_dataset = dataset.CustomDataset_liver(const.liver_path_valid, transform=image_transform)
test_dataset = dataset.CustomDataset_liver(const.liver_path_test, transform=image_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Plot the images from the liver dataset
plt.figure(figsize=(10, 5))
plt.suptitle('Liver Dataset')

for i, (x, y) in enumerate(train_loader):
    for j in range(4):
        plt.subplot(2, 4, i * 4 + j + 1)
        plt.imshow(x[j][0], cmap='gray')
        plt.title(f'Image {i * 4 + j + 1}')
        plt.axis('off')
    if i == 1:  # Display only first batch
        break

plt.tight_layout()
plt.show()

num_images_to_show = 5
for i in range(num_images_to_show):
    image, mask = train_dataset[i]  # Получить i-ое изображение и его маску из набора данных
    plt.figure(figsize=(8, 4))

    # Отобразить оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Отобразить маску
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.show()

################################## MODEL ###################################

model = model_Unet.UNet(num_classes=2)

# loss fun and optimizer
loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

early_stopping = model_original.EarlyStopping(patience = 10,delta = 0.)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if const.fine_tuning:
    model.load_state_dict(torch.load(const.model_in))

if const.test:
    model.load_state_dict(torch.load(const.model_test))

    test_loss, test_accuracy = model_Unet.test_model(model, test_loader, loss_fn, device = device)
    model_Unet.visualize_predictions(model, test_loader, device)

else:

    model_Unet.train_model(model, train_loader, valid_loader, loss_fn, optimizer, num_epochs=10, device = device)
    torch.save(model.state_dict(), 'model_Unet.pth')



