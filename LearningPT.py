# Learning the basics

# Working with Data



def working_with_data():

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda, Compose
    import matplotlib.pyplot as plt

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

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]:", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
            )
        
        def forward(self, x):
            x  = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad(): # Don't calculate the gradient
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer)
        test(test_dataloader, model)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

def Tensors():
    import torch
    import numpy as np

    data = [[1,2],[3,4]]
    x_data = torch.tensor(data)

    # From array to torch tensor
    """
        np_array = np.array(data)
        x_np = torch.from_numpy(np_array)
    """

    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor: \n {x_rand} \n")

    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    tensor = torch.rand(3,4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

    tensor = torch.ones(4,4)
    print('First row: ',tensor[0])
    print('First column: ', tensor[:, 0])
    print('Last column:', tensor[..., -1])
    tensor[:,1] = 0
    print(tensor)

    # Joining tensors
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # Arithmetic Operations

    # Different ways to calculate matrix multiplication
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)

    # Element-wise product
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

    # Element-wise tensors
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))

    # In-place operations
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

def bridge_tensor():
    import torch
    import numpy as np

    # Tensor to Numpy array
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    # If the original torch tensor changes, the numpy array changes
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    # Numpy array to Tensor
    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")

    pass

# Creating a Custom dataset for your files
def CustomImageDataset(Dataset):
    import os
    import pandas as pd
    from torchvision.io import read_image

    class CustomDataset(Dataset):
        
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
        
        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            sample = {"image": image, "label": label}
            return sample

def transforms(self):
    import torch
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda

    """
    Data does not always come in its final processed form that is 
    required for training machine learning algorithms. We use 
    transforms to perform some manipulation of the data and make 
    it suitable for training.

    All TorchVision datasets have two parameters -transform to 
    modify the features and target_transform to modify the labels 
    - that accept callables containing the transformation logic. 
    The torchvision.transforms module offers several commonly-used 
    transforms out of the box.

    The FashionMNIST features are in PIL Image format, and the 
    labels are integers. For training, we need the features as 
    normalized tensors, and the labels as one-hot encoded tensors. 
    To make these transformations, we use ToTensor and Lambda.
    """
    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

def build_neural_network():
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
    
    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    input_image = torch.rand(3, 28, 28)
    print(input_image.size())

    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())

    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())

    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)

    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)

    print("Model structure: ", model, "\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

def auto_grad():
    import torch

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    print('Gradient function for z =',z.grad_fn)
    print('Gradient function for loss =', loss.grad_fn)

    loss.backward() #calculates the gradients for layers
    print(w.grad)
    print(b.grad)

    # Disable gradient
    z = torch.matmul(x, w)+b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print(z.requires_grad)

    z = torch.matmul(x, w)+b
    z_det = z.detach()
    print(z_det.requires_grad)

    inp = torch.eye(5, requires_grad=True)
    out = (inp+1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("First call\n", inp.grad)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("\nSecond call\n", inp.grad)
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("\nCall after zeroing gradients\n", inp.grad)

def optimizing_params():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda

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

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.Softmax()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork()

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    """
    UNDERSTANDING AND CREATING THE OPTIMIZATION LOOP
    """

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Set the optimizer
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
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    #working_with_data()
    #Tensors()
    #bridge_tensor()
    #build_neural_network()
    #auto_grad()
    optimizing_params()