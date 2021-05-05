from torch import cuda, device
from time import time


GPU = "0"  # 0: 2080TI;  1: P2000
dev = device("cuda:" + GPU if cuda.is_available() else "cpu")
info = f"Executing via: {dev}"
tbeg = time()


def game_file(n):
    # Recursive Fibonacci as a placeholder for a game
    #
    if n <= 1:
        return n
    else:
        return (game_file(n - 1) + game_file(n - 2))


tend = time()

time_to_calc = tend - tbeg
