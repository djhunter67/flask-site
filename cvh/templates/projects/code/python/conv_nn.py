# Dogs andCats Convolutional NN
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from colorama import Fore as F


R = F.RESET
# set to true once,
# then back to false unless you want to change something in your training data
REBUILD_DATA = False
VAL_PCT = 0.1  # lets reserve 10% of our data for validation
BATCH_SIZE = 500
EPOCHS = 10
DEV = "cuda:0"

if torch.cuda.is_available():
    device = torch.device(DEV)
    print(f"running on {DEV}")
else:
    device = torch.device("cpu")
    print("running on cpu")


def main():
    if REBUILD_DATA:
        dogs_vs_cats = DogsVSCats()
        dogs_vs_cats.make_training_data()        

    
    tbeg = time.time()
    tbeg2 = time.time()
    net = Net().to(device)
    

    training_data = np.load("training_data.npy", allow_pickle=True)
    global X, y, val_size
    
    X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X / 255.0
    y = torch.Tensor([i[1] for i in training_data])
    tend2 = time.time()
    print(f"{F.CYAN}Constructor & Load Data Time:{R} {F.RED}{round(tend2 - tbeg2, 5)}{R}")
    val_size = int(len(X) * VAL_PCT)
    
    train(net)
    test(net)
    tend = time.time()
    print(f"{F.RED}TIME TO COMPUTE ON{R} {F.GREEN}{device}{R}: {round(tend - tbeg, 4) / 60} minutes")

class DogsVSCats():
    """Pull in the Dog and cats pictures"""
    IMG_SIZE = 50
    CATS = "PetImages\Cat"
    DOGS = "PetImages\Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append(
                        [np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print(f"\nCats: {self.cat_count}")
        print(f"\nDogs: {self.dog_count}")


class Net(nn.Module):
    """Just run the init of parent class (nn.Module)"""

    def __init__(self):
        """init method of the inhgerited class"""
        super().__init__()
        # Input is 1 image, 32 output channels, 5x5 kernel / window
        #
        self.convolution_nueral_layer1 = nn.Conv2d(1, 32, 5)
        # Input is 32, bc the first layer output 32.
        # Then we say the output will be 64 channels, 5x5 conv
        self.convolution_nueral_layer2 = nn.Conv2d(32, 64, 5)
        self.convolution_nueral_layer3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        # Flattening
        #
        self.fullyconnected1 = nn.Linear(self._to_linear, 512)
        # 512 in, 2 out bc we're doing 2 classes (dog vs cat)
        #
        self.fullyconnected2 = nn.Linear(512, 2)

    def convs(self, x):
        """Figuring out the flattened data shape"""
        # Max pooling over 2x2
        x = Fn.max_pool2d(Fn.relu(self.convolution_nueral_layer1(x)), (2, 2))
        x = Fn.max_pool2d(Fn.relu(self.convolution_nueral_layer2(x)), (2, 2))
        x = Fn.max_pool2d(Fn.relu(self.convolution_nueral_layer3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flattens X before
        #
        x = x.view(-1, self._to_linear)
        x = Fn.relu(self.fullyconnected1(x))
        # This is our output layer. No activation here
        #
        x = self.fullyconnected2(x)

        return Fn.softmax(x, dim=1)


def train(net):

    train_X = X[:-val_size]
    train_y = y[:-val_size]
    time_sum = 0
    loss_sum = 0

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        tbeg1 = time.time()
        # from 0, to the len of x,
        # stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        for i in range(0, len(train_X), BATCH_SIZE):
            # print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCH_SIZE]

           
            batch_X = batch_X.to(device) 
            batch_y = batch_y.to(device)

            # Zero the gradient
            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            # Does the update
            #
            optimizer.step()
        tend1 = time.time()
        time_sum += tend1 - tbeg1
        loss_sum += loss
        print(f"{F.CYAN}EPOCH:{R} {epoch} Loss: {F.RED}{loss:.5f}{R}")        
        print(f"{F.CYAN}EPOCH:{R} {epoch} {F.GREEN}{round(tend1 - tbeg1, 3)}{R} secs\n")
    print(f"{F.YELLOW}AVERAGE LOSS:{R} {F.GREEN}{loss_sum / EPOCHS}{R}")
    print(f"{F.YELLOW}AVERAGE TIME:{R} {F.GREEN}{time_sum / EPOCHS}{R}")


def test(net):

    test_X = X[-val_size:]
    test_y = y[-val_size:]
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))


if __name__ == '__main__':
    main()
