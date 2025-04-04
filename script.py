from metric_collector import iterate_configs

# Create configurations for the experiments
num_runs = 1 # 10
tmax = 2 # 50
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


if __name__ == "__main__":
    iterate_configs(configs0, num_runs, tmax)

