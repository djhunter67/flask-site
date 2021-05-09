import csv, time
from pandas.core.frame import DataFrame
from torch import (
    cuda,
    from_numpy,
    device,
    optim,
    tanh,
    softmax,
    FloatTensor,
    LongTensor,
    nn,
    save,
    load,
    autograd,
)
import time
from os import path
from tqdm import tqdm
from colorama import Fore as F
from pandas import read_csv
from numpy import array, float16
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


R = F.RESET
TEST_PERC = 0.333
RAND_STATE = 0
GPU = "0"  # 0: 2080TI;  1: P2000

dev = device("cuda:" + GPU if cuda.is_available() else "cpu")


def main():
    """3 LAYER FWD NEURAL NETWORK"""
    display_header("NUMERAI w/ TORCH")

    print(f"Executing via: {dev}")

    # Load entire dataset onto db on first run
    #
    if path.isfile("tensor_data.pt"):
        print(f"Reading from db")
        tbeg1 = time.time()
        # Load from tensor object onto the GPU or CPU
        #
        data = load("tensor_data.pt", map_location=lambda storage, loc: dev)
        tend1 = time.time()
        print(f"Time to read data: {F.YELLOW}{(tend1 - tbeg1):.3f}{R} secs")
    else:
        print(f"\nReading from file")
        tbeg1 = time.time()
        save(read_file("numerai_training_data.csv"), "tensor_data.pt")
        data = load("tensor_data.pt", map_location=lambda storage, loc: dev)
        tend1 = time.time()
        print(f"Time to read data: {F.YELLOW}{(tend1 - tbeg1):.3f}{R} secs")

    #######################################################################
    # PREPARING DATA                                                      #
    #######################################################################

    print(f"Preparing data")
    tbeg2 = time.time()
    X_train_std, y_train, X_test_std, y_test = prepare_data(data)

    X_train_std = from_numpy(X_train_std).type(FloatTensor).to(dev)
    X_test_std = from_numpy(X_test_std).type(FloatTensor).to(dev)
    y_train = from_numpy(y_train).type(LongTensor).to(dev)
    y_test = from_numpy(y_test).type(LongTensor).to(dev)
    tend2 = time.time()
    print(f"Time to prepare data: {F.YELLOW}{(tend2 - tbeg2):.3f}{R} secs")

    ######################################################################
    # INSTANTIATE MODEL                                                  #
    ######################################################################
     
    input_size = X_train_std.size()[1]
    hidden_size = input_size // 2
    num_classes = 2

    model = System(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(dev)
    optimize = optim.Adam(params=model.parameters(), lr=0.001)

    #######################################################################
    # TRAIN & TEST                                                        #
    #######################################################################

    train_loss, training_time = train(autograd.Variable(X_train_std),
                                      autograd.Variable(y_train), 
                                      model,
                                      optimize,
                                      6001)
    test_loss, testing_time = test(autograd.Variable(X_test_std), 
                                   autograd.Variable(y_test), 
                                   model, 
                                   optimize, 
                                   600)

    #######################################################################
    # RESULTS                                                             #
    #######################################################################

    print(
        f"\nTRAINING: {F.RED}{abs(train_loss + 1):.5f}{R} ERROR @ {F.GREEN}{training_time:.4f}{R} secs"
    )
    print(
        f"TESTING: {F.RED}{abs(test_loss + 1):.5f}{R} ERROR @ {F.GREEN}{testing_time:.4f}{R} secs"
    )

    percentage = (abs(train_loss - test_loss) / (train_loss + test_loss) / 2) * 100

    if abs(percentage) > 0.15:
        print(f"\n\nOVERFIT: {F.RED}{abs(percentage):.4f}{R}\n\n")
    else:
        print(f"\nOVERFIT: {F.GREEN}{abs(percentage):.4f}{R}")
        # print(f"ACCURACY: {F.GREEN}{abs(percentage) - 1:.4f}{R}\n\n")

    if dev.type != "cpu":
        print(cuda.get_device_name(int(GPU)))
        print("Memory Usage:")
        print("Allocated:", round(cuda.memory_allocated(int(GPU)) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(cuda.memory_reserved(int(GPU)) / 1024 ** 3, 1), "GB")


def display_header(header) -> str:
    """Succint description of the program"""
    print(f"\n{F.GREEN}------------------------------------------")
    print(f"            {header}")
    print(f"------------------------------------------{R}\n")


def read_file(filename) -> DataFrame:
    # cols_list = [x + 3 for x in range(len(filename))]
    cols_list = []

    with open(filename) as fin:
        column_names = next(csv.reader(fin))
        cols_list = len(column_names)

    cols_list = list(range(3, cols_list))
    dtypes = {x: float16 for x in column_names if x.startswith(("feature", "target"))}
    df = read_csv(
        filename, dtype=dtypes, usecols=cols_list, header=0)#, nrows=10_000)

    return df


def prepare_data(data):

    X = array(data.iloc[:, :-1])
    y = array(data.iloc[:, -1])

    # split the problem into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERC, random_state=RAND_STATE
    )

    # scale X
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test


class System(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 4)
        self.layer_3 = nn.Linear(hidden_size // 4, num_classes)

    # 3 Layers
    def forward(self, x):
        x = self.layer_1(x)
        x = tanh(x)
        x = self.layer_2(x)
        x = tanh(x)
        x = self.layer_3(x)
        x = softmax(x, dim=-1)
        return x


def train(input, target, model, optimize, epochs):
    
    train_losses = 0
    start_time = time.time()
    for _ in tqdm(range(epochs), desc="TRAINING"):

        # Forward propagation
        out = model(input).to(dev)
        l1_loss = nn.NLLLoss()
        loss = l1_loss(out, target).to(dev)

        model.zero_grad()
        # Backward propagation
        loss.backward()
        optimize.step()
        train_losses = loss
        #if _ % 300 == 0:
            #print(f"\n{F.YELLOW}training iteration {_}{R}\nLOSS: {F.RED}{loss}{R}\n")

    training_time = time.time() - start_time

    return train_losses.item(), training_time


def test(input, target, model, optimize, epochs):

    #results = []
    losses = 0
    start_time = time.time()
    for _ in tqdm(range(epochs), desc="TESTING"):

        # Forward propagation
        out = model(input).to(dev)
        l1_loss = nn.NLLLoss()
        loss = l1_loss(out, target).to(dev)

        model.zero_grad()
        # Backward propagation
        loss.backward()
        optimize.step()
        losses = loss
        #results = out
    testing_time = time.time() - start_time
    #print(results)
    return losses.item(), testing_time


if __name__ == "__main__":
    main()