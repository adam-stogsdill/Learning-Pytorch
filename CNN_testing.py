import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(training_data, batch_size=64)
testing_loader = DataLoader(test_data, batch_size=64)

class CNN_Model(nn.Module):

    def __init__(self, name):
        super(CNN_Model, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 16, kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(11664, 512)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(512, 10)
        self.output = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        return self.output(x)

model = CNN_Model(name="CNN MODEL")

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad() # Zero out the optimizer gradient
        loss.backward()       # Calculate the gradient
        optimizer.step()      # Optimizer steps through gradient to update parameters

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # get the sum of correct predictions

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(testing_loader, model, loss_fn)
print("Done!")