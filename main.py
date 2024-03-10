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

# We define a function to visualize the evolution of the loss and the metric.
def loss_and_metric_plot(results:dict):

    plt.style.use('ggplot')
    training_loss = results['train_loss']

    validation_loss = results['val_loss']

    fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3.8))
    ax = ax.flat

    ax[0].plot(training_loss, 'o-', markersize = 4, label = "Train")
    ax[0].plot(validation_loss, label = "Val")
    ax[0].set_title("BinaryLoss", fontsize = 12, fontweight = "bold", color = "black")
    ax[0].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[0].set_ylabel("loss", fontsize = 10, fontweight = "bold", color = "black")

    ax[0].legend()
    fig.show()
    fig.savefig('loss_plot.png')


# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.Grayscale(num_output_channels=3),  # Преобразовать в градации серого с тремя каналами
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Предполагая серое изображение
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

custom_dataset = dataset.CustomDataset(const.path_valid)#, transform=image_transform)
image, mask = custom_dataset[0]
print(image.shape, mask.shape)


fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i in range(3):  # Plot the first three samples
    image, mask = custom_dataset[i]
    axs[i, 0].imshow(image)
    axs[i, 0].set_title('Image')
    axs[i, 0].axis('off')
    axs[i, 1].imshow(mask, cmap='gray')
    axs[i, 1].set_title('Mask')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Create datasets
train_dataset = dataset.CustomDataset_general(const.path_train, transform=image_transform)
valid_dataset = dataset.CustomDataset_general(const.path_valid, transform=image_transform)
test_dataset = dataset.CustomDataset_general(const.path_test, transform=image_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))

'''# Count number of ones and zeros in tensor x
num_ones = torch.eq(x, 1).sum().item()
num_zeros = x.numel() - num_ones

#print("Number of ones in x:", num_ones)
#print("Number of zeros in x:", num_zeros)

# Count number of ones and zeros in tensor y
num_ones = torch.eq(y, 1).sum().item()
num_zeros = y.numel() - num_ones

#print("Number of ones in y:", num_ones)
#print("Number of zeros in y:", num_zeros)'''

# Convert tensors to numpy arrays and squeeze the channel dimension
print(x.numpy().shape)
x_np = x.numpy().squeeze(1)
y_np = y.numpy().squeeze(1)

#Plot the images
plt.figure(figsize=(10, 5))

#Plot original images
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_np[i], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

# Plot masks
for i in range(4):
    plt.subplot(2, 4, i + 5)
    plt.imshow(y_np[i], cmap='gray')
    plt.title('Mask')
    plt.axis('off')

plt.tight_layout()
plt.show()

#dataset.display_images(test_dataset)

C = 1 #How many channels are in the input?
n_filters = 32 #Smallest value of filters you should usually consider. If we wanted to try and optimize the architecture we could use Optuna to pick a better number of filters.
loss_func = nn.BCEWithLogitsLoss() #BCE loss implicitly assumes a binary problem


#Original model
'''model = nn.Sequential(
    model_original.cnnLayer(C, n_filters), #First layer changes number of channels up to the large numer
    *[model_original.cnnLayer(n_filters, n_filters) for _ in range(5)], #Create 5 more hidden layers
    #Make a prediction for _every_ location. Notice we use 1 channel out, since we have a binary problem and are using BCEWithLogitsLoss as our loss function.
    nn.Conv2d(n_filters, 1, (3,3), padding=1), #Shape is now (1, W, H)
)'''

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

    #model_original.test(model, test_loader, loss_fn)
    #model_original.visualize_predictions(model, test_loader)
    test_loss, test_accuracy = model_Unet.test_model(model, test_loader, loss_fn, device = device)
    model_Unet.visualize_predictions(model, test_loader, device)

else:
    '''results = model_original.Train(model.to(device),
                train_dataloader=train_loader,
                val_dataloader=valid_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                early_stopping=early_stopping, lr_scheduler = lr_scheduler,
                epochs=const.epochs,
                device=device)

    loss_and_metric_plot(results)'''

    model_Unet.train_model(model, train_loader, valid_loader, loss_fn, optimizer, num_epochs=10, device = device)
    torch.save(model.state_dict(), 'model_Unet.pth')



