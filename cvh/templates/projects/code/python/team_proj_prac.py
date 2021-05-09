# Christerpher Hunter
# EEE 498: ML w/ FPGA's
# Edward and Cristen
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fn
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from colorama import Fore as F


R = F.RESET
DEV = "cuda:0"

if torch.cuda.is_available():
    device = torch.device(DEV)
    print(f"running on the {DEV}")
else:
    device = torch.device("cpu")
    print("running on cpu")


def main():

    tbeg = time.time()
    train = datasets.MNIST("",
                           train=True,
                           download=False,
                           transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("",
                          train=False,
                          download=False,
                          transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    total, correct, net, X = execute(trainset, 2, "TRAINING COMPLETE")
    checks(total, correct, net, X, tbeg)
    total, correct, net, X = execute(testset, 2, "TEST COMPLETE")
    checks(total, correct, net, X, tbeg)

def checks(total, correct, net, X, tbeg):

    print(f'{F.GREEN}Accuracy: {round(correct / total, 3)}{R}')
    tend = time.time()
    print(f"{F.RED}TIME TO COMPLETION: {R}{F.GREEN}{(round(tend - tbeg, 4) / 60):.2f}{R} minutes")

    #tbeg1 = time.time()
    print(f'\nExpected Number:\
    {F.LIGHTBLUE_EX}{torch.argmax(net(X[0].view(-1, 28 * 28))[0])}{R}')
    
    #tend1 = time.time()
    #print(f"\n{F.MAGENTA}Time to get expected number:{R} {round(tend1 - tbeg1, 7)} secs")

    X = X.to("cpu")
    plt.imshow(X[0].view(28, 28))
    plt.show()


def execute(test_or_train, EPOCHS=5, status=f"\n"):
    tbeg4 = time.time()
    for data in test_or_train:
        # print(data)
        continue

    X, y = data[0][0], data[1][0]
    # print(f'\n{y}')

    layer_1_weights = 0.0
    layer_2_weights = 0.0
    layer_3_weights = 0.0
    layer_4_weights = 0.0

    net = Net().to(device)

    X = torch.rand((28, 28))
    X = X.view(-1, 28*28)
    X = X.to(device)

    output = net(X).to(device)
    tend4 = time.time()
    print(
        f"\n\n{F.CYAN}Constructors & grab Dataset time:{R} {round(tend4 - tbeg4, 4)}")

    # Parameters refers to everything adjustable
    #
    # Learning rate set, non-decaying
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    time_sum = 0
    loss_sum = 0

    for _ in range(EPOCHS):  # whole passes through the dataset
        tbeg3 = time.time()
        for data in test_or_train:
            # data is a batch of feature set and lables
            X, y = data
            X = X.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net(X.view(-1, 28 * 28))

            # Calculate loss for One Hot Vector ex. [0, 0, 1, 0, 0]
            loss = Fn.nll_loss(output, y)
            # Back propogation of the weights
            #
            loss.backward()
            optimizer.step()  # Adjust the weights
        print(
            f'\nLOSS: {F.RED}{loss:.9f}{R} Iteration # {F.GREEN}{_ + 1}{R} of {EPOCHS}')
        tend3 = time.time()

        time_sum += tend3 - tbeg3
        loss_sum += loss
        print(f"{F.CYAN}Time This round:{R} {round(tend3 - tbeg3, 4)} secs")
    
    print(f"\n{F.YELLOW}AVERAGE LOSS:{R} {F.GREEN}{(loss_sum / EPOCHS):.3f}{R}")
    print(f"{F.YELLOW}AVERAGE TIME:{R} {F.GREEN}{(time_sum / EPOCHS):.3f}{R}\n")

    if status == "TRAINING COMPLETE":
        layer_1_weights = torch.sum(net.full_connected_1.weight.data)
        layer_2_weights = torch.sum(net.full_connected_2.weight.data)
        layer_3_weights = torch.sum(net.full_connected_3.weight.data)
        layer_4_weights = torch.sum(net.full_connected_4.weight.data)
        print(f"WEIGHTS, layer 1: {layer_1_weights}")
        print(f"WEIGHTS, layer 2: {layer_2_weights}")
        print(f"WEIGHTS, layer 3: {layer_3_weights}")
        print(f"WEIGHTS, layer 4: {layer_4_weights}")
    
    print(f"{F.GREEN}{status}{R}\n")

    correct = 0
    total = 0
    # How good is the network at this point
    #
    tbeg5 = time.time()
    with torch.no_grad():
        for data in test_or_train:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output = net(X.view(-1, 28 * 28))
            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct += 1
                total += 1
    tend5 = time.time()
    print(f"{F.MAGENTA}Time to calculate accuracy:{R} {round(tend5 - tbeg5, 3)} secs")    
    return total, correct, net, X


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # THE NEURAL NET LAYERS
        #
        # Pass in the flattened image to rows of a 64 neuron hidden layer
        self.full_connected_1 = nn.Linear(28 * 28, 64)
        # Take in the previous output and output whatever
        self.full_connected_2 = nn.Linear(64, 64)
        self.full_connected_3 = nn.Linear(64, 64)
        self.full_connected_4 = nn.Linear(64, 10)  # output the 10 length array

    # Pass data in one direction
    #
    def forward(self, x):
        # relu is the activation function
        x = Fn.relu(self.full_connected_1(x))
        x = Fn.relu(self.full_connected_2(x))
        x = Fn.relu(self.full_connected_3(x))
        x = self.full_connected_4(x)

        return Fn.log_softmax(x, dim=1)


if __name__ == "__main__":
    main()
