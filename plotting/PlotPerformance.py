import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from matplotlib import pyplot as plt

def check(num: str, name: str):

    try:
        num = int(num)
    except:
        raise argparse.ArgumentTypeError(f"The {name} must be an integer.")
    

    if num <= 0:
        raise argparse.ArgumentTypeError(f"The {name} must be positive and greater than zero")
    
    return num

def checkMode(mode: str):

    try:
        mode = int(mode)
    except:
        raise argparse.ArgumentTypeError(f"The mode must be an integer.")
    
    if mode != 0 and mode != 1:
        raise argparse.ArgumentTypeError(f"The mode must be 0 or 1")
    
    return mode

def main():

    #
    # Set up ArgumentParser
    #

    parser = argparse.ArgumentParser(description="Visualize the performance of the Monte Carlo Tree Search algorithm given its step number in form of a boxplot.")
    
    parser.add_argument("--start", help="Set the start number of MCTS steps.", type=lambda start: check(start, name="step number"), required=True)
    parser.add_argument("--stop", help="Set the end number of MCTS steps (inclusive).", type=lambda stop: check(stop, name="step number"), required=True)
    parser.add_argument("--step", help="Set the step number going from start to end number of MCTS steps.", type=lambda step: check(step, name="step number"), required=True)
    parser.add_argument("--mode", help="Set the type of the MCTS algorithm: 0 = MCTS_ValueOnly, 1 = MCTS_Reward.", type=lambda mode: checkMode(mode), required=True)

    args = parser.parse_args()

    start = args.start 
    stop = args.stop 
    step = args.step
    mcts_mode = args.mode

    #
    # Run
    #
    
    if mcts_mode == 0:
        mcts_mode = "MCTS_ValueOnly"
    else:
        mcts_mode = "MCTS_Reward"

    sns.set_theme(style="ticks", palette="pastel")
    
    num_mcts_iterations = range(start, stop + step, step)

    dfs = []
    for i in num_mcts_iterations:

        x = np.load(f"../logs/{mcts_mode}/num_steps_{i}.npy")

        data = {"Number of MCTS iterations": i, 
                    "Number of steps": x}

        df = pd.DataFrame(data)

        dfs.append(df)

    dfs = pd.concat(dfs)
    print(dfs)
    sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Number of steps")
    sns.despine(offset=10, trim=True)

    plt.grid(True)
    plt.title(mcts_mode)
    plt.tight_layout()
    plt.savefig(f"./plots/{mcts_mode}_Performance.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
