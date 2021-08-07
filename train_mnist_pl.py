import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import transforms

from torch.nn import functional as F

# Step 1 model definition
# Pytorch Lightning
class MNISTClassifier(pl.LightningModule):
  
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

  # Step 3 optimizer
  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
      return optimizer

  # Step 4 loss function
  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  # Step 5 training loop
  def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)   # we already defined forward and loss in the lightning module. We'll show the full code next
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

  def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

#step 2 data
class MNISTDataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
      
    # prepare transforms standard to MNIST
    self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    
    self.mnist_train, self.mnist_val = random_split(self.mnist_train, [55000, 5000])

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=64)

  def val_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=64)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=64)


  # train
model = MNISTClassifier()
data_module = MNISTDataModule()
trainer = pl.Trainer()

trainer.fit(model, data_module)