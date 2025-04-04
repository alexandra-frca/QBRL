from concurrent.futures import ProcessPoolExecutor
from metric_collector import run_config
from src.utils import product_dict
from tqdm import tqdm
from pprint import pprint
import sys, os
import time

# Create configurations for the experiments
num_runs = 1
tmax = 1
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
    return run_config(config, num_runs, tmax)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    mw = 2
    print(f"> Max workers = {mw}.")
    print("> Configs:")
    pprint(configs0)
    print("__________________")

if __name__ == "__main__":
    ti = time.time()

    # Extract results from multiple runs in parallel
    mw = 4
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

    tf = time.time()
    dt = tf - ti
    hours, rem = divmod(dt, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    files.download('results.csv')

