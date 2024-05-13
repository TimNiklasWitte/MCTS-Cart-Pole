import gymnasium
import numpy as np
import tqdm
import argparse
from multiprocessing import Process, shared_memory

from MCTS_ValueOnly import *
from MCTS_Reward import *


# openai gym causes a warning - disable it
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


def process_runMCTS(num_procs, process_id, seed, mcts_mode, num_mcts_iterations, num_steps_shm):
    
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=seed)

    num_steps_mem = np.ndarray(shape=(num_procs, ), dtype=np.uint16, buffer=num_steps_shm.buf)

    iterator = range(1000)
    if process_id == 0:
        iterator = tqdm.tqdm(iterator, position=0, leave=True)

    for step_cnt in iterator:

        if mcts_mode == 0:
            mcts = MCTS_ValueOnly(env)
        else: 
            mcts = MCTS_Reward(env)

        best_action = mcts.run(num_mcts_iterations)
        state, reward, done, _, _  = env.step(best_action)

        if done:
            break

   
    num_steps_mem[process_id] = step_cnt


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

    parser = argparse.ArgumentParser(description="Let the Monte Carlo Tree Search algorithm play multiple episodes in the Cart Pole environment concurrently.")
    
    parser.add_argument("--start", help="Set the start number of MCTS steps.", type=lambda start: check(start, name="step number"), required=True)
    parser.add_argument("--stop", help="Set the end number of MCTS steps (inclusive).", type=lambda stop: check(stop, name="step number"), required=True)
    parser.add_argument("--step", help="Set the step number going from start to end number of MCTS steps.", type=lambda step: check(step, name="step number"), required=True)
    parser.add_argument("--procs", help="Set the number of processes.", type=lambda procs: check(procs, name="number of processes"), required=True)
    parser.add_argument("--mode", help="Set the type of the MCTS algorithm: 0 = MCTS_ValueOnly, 1 = MCTS_Reward.", type=lambda mode: checkMode(mode), required=True)

    args = parser.parse_args()

    start = args.start 
    stop = args.stop 
    step = args.step
    num_procs = args.procs
    mcts_mode = args.mode

    #
    # Run
    #

    for num_mcts_iterations in range(start, stop + step, step):
        
        print(num_mcts_iterations)

        num_steps_mem = np.zeros(shape=(num_procs,), dtype=np.uint16)
   
        num_steps_shm = shared_memory.SharedMemory(create=True, size=num_steps_mem.nbytes)
    
        procs_list = []
        for process_id in range(num_procs):
            
            seed = np.random.randint(0, 99999999)
            proc = Process(target=process_runMCTS, 
                           args=(num_procs, process_id, seed, mcts_mode, num_mcts_iterations, num_steps_shm,))
            procs_list.append(proc)
            proc.start()


        # complete the processes
        for proc in procs_list:
            proc.join()
    
        num_steps_mem = np.ndarray(shape=(num_procs, ), dtype=np.uint16, buffer=num_steps_shm.buf)

        if mcts_mode == 0:
            dir_name = "MCTS_ValueOnly"
        else:
            dir_name = "MCTS_Reward"

        np.save(f"./logs/{dir_name}/num_steps_{num_mcts_iterations}.npy", num_steps_mem)

        num_steps_shm.close()
        num_steps_shm.unlink()

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")