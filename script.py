from concurrent.futures import ProcessPoolExecutor
from metric_collector import run_config
from src.utils import product_dict
from tqdm import tqdm
from pprint import pprint
import sys, os

# Create configurations for the experiments
num_runs = 1 #40 
time = 5 # 50
configs0 = {
    "experiment": ["tiger", "robot", "gridworld"],
    "discount": [0.9],
    "horizon": [2],
    "c_sample": [5, 15, 50, 100],
    "r_sample": [250]
}

configs0 = {
    "experiment": ["tiger"],
    "discount": [0.9],
    "horizon": [2],
    "c_sample": [5],
    "r_sample": [250]
}

# Create list of dictionaries as product of dictionary of lists
configs = list(product_dict(configs0))

# Create iterator function
def foo(config):
    return run_config(config, num_runs, time)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    mw = 1
    print(f"> Max workers = {mw}.")
    print("> Configs:")
    pprint(configs0)
    print("__________________")

    with ProcessPoolExecutor(max_workers=mw) as executor:
        try:
            # Iterate each config
            _ = list(tqdm(executor.map(foo, configs),
                          total=len(configs), 
                          desc="Iterating configs", 
                          position=0, 
                          leave=False))
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            print('KeyboardInterrupt 1')
                        
