import torch
from torch import nn

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import transforms

from torch.nn import functional as F

# Step 1 model definition
# Standard pytorch module
class MNISTClassifier(nn.Module):
  
  def __init__(self):
    super(MNISTClassifier, self).__init__()
    # mnist images are (1, 28, 28) (channels, width, height) 
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, width, height = x.size()
    # (b, 1, 28, 28) -> (b, 1*28*28)
    x = x.view(batch_size, -1)

    # layer 1
    x = self.layer_1(x)
    x = torch.relu(x)

    # layer 2
    x = self.layer_2(x)
    x = torch.relu(x)

    # layer 3
    x = self.layer_3(x)

    # probability distribution over labels
    x = torch.log_softmax(x, dim=1)

    return x

# Step 2 Data loading and transforming
# ----------------
# TRANSFORMS
# ----------------
# prepare transforms standard to MNIST
transform=transforms.Compose([transforms.ToTensor(), 
                              transforms.Normalize((0.1307,), (0.3081,))])

# this has been adjusted to not have everyone download the data over and over during the workshop
# Yann LeCunn's website is known to get a DoS and return http code 503
# if you want to actually download this replace with:
# mnist_source = os.getcwd()
mnist_source = "/home/azureuser/cloudfiles/code"

# ----------------
# TRAINING, VAL DATA
# ----------------
mnist_train = MNIST(mnist_source, train=True, transform=transform, download=True)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

# ----------------
# TEST DATA
# ----------------
mnist_test = MNIST(mnist_source , train=False, download=True)

# ----------------
# DATALOADERS
# ----------------
# The dataloaders handle shuffling, batching, etc...
mnist_train = DataLoader(mnist_train, batch_size=64)
mnist_val = DataLoader(mnist_val, batch_size=64)
mnist_test = DataLoader(mnist_test, batch_size=64)

# Step 3 Setup Optimizer
pytorch_model = MNISTClassifier()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=1e-3)

# Step 4 Loss function
def cross_entropy_loss(logits, labels):
  return F.nll_loss(logits, labels)


# Step 5 Training loop
# standard PyTorch training loop looks something like this:
# ----------------
# TRAINING LOOP
# ----------------
num_epochs = 3
for epoch in range(num_epochs):

  # TRAINING LOOP
  for train_batch in mnist_train:
    x, y = train_batch

    logits = pytorch_model(x)
    loss = cross_entropy_loss(logits, y)
    print('train loss: ', loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

  # VALIDATION LOOP
  with torch.no_grad():
    val_loss = []
    for val_batch in mnist_val:
      x, y = val_batch
      logits = pytorch_model(x)
      val_loss.append(cross_entropy_loss(logits, y).item())

    val_loss = torch.mean(torch.tensor(val_loss))
    print('val_loss: ', val_loss.item())