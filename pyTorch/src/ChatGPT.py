import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_module_stereo import Flickr1024

# Define hyperparameters
lr = 0.001
batch_size = 32
num_epochs = 10
image_size = 100
upscale_factor = 2

# Load CIFAR-10 dataset
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor()
# ])

train_dir = 'C:/Users/Percy/Flickr1024/Train'
test_dir = 'C:/Users/Percy/Flickr1024/Test'
val_dir = 'C:/Users/Percy/Flickr1024/Validation'

trainset = Flickr1024(root_dir=train_dir, im_size=image_size, scale=upscale_factor, transform=transforms)
validset = Flickr1024(root_dir=val_dir, im_size=image_size, scale=upscale_factor, transform=transforms)

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
validloader = DataLoader(validset, batch_size=8, shuffle=True)

# Define vision transformer model
vit = ViT(
    image_size=image_size,
    patch_size=4,
    num_classes=10,
    dim=128,
    depth=6,
    heads=8,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1
)

# Define super resolution model
class SuperResolution(nn.Module):
    def __init__(self, upscale_factor):
        super(SuperResolution, self).__init__()
        self.upscale_factor = upscale_factor
        self.vit = vit
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16 * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = self.vit(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(superres.parameters(), lr=lr)

# Train super resolution model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = nn.functional.interpolate(inputs, scale_factor=upscale_factor, mode='bicubic')
        optimizer.zero_grad()
        outputs = superres(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
print('Finished Training')
