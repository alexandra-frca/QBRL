from src.rl_algorithms.pomdp_lookahead import build_tree, pomdp_lookahead
from get_ddns import get_tiger_ddn, get_robot_ddn, get_gridworld_ddn
from src.utils import get_avg_reward_and_std, belief_update, product_dict
from src.networks.qbn import QuantumBayesianNetwork as QBN
from src.networks.bn import BayesianNetwork as BN
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
import os, pytz
from datetime import datetime
from tqdm import tqdm
import  os
import time
from datetime import datetime


def get_tree(ddn, horizon):
    action_space = ddn.get_space(ddn.action_type)
    observation_space = ddn.get_space(ddn.observation_type)
    tree = build_tree({}, action_space, observation_space, horizon)
    return tree


def get_sample_coefficients(ddn, tree, belief_state, n_samples, quantum=False):
    r = 0
    
    # Iterate all action nodes
    for action_tree in tree.children:
        # Iterate all observation nodes:
        action = action_tree.attributes["action"]
        
        # Add to results
        if len(action_tree.children) > 0:
            observation_nodes = ddn.get_nodes_by_type(ddn.observation_type)
            evidence = {**action, **belief_state}
            probs = ddn.query(observation_nodes, evidence, n_samples)[["Prob"]].values
            if quantum:
                r += (np.sqrt(1 / (probs + 1e-6))).sum()
            else:
                r += (1 / (probs + 1e-6)).sum()
        
        # Recursive call
        for observation_tree in action_tree.children:
            observation = observation_tree.attributes["observation"]
            new_belief = belief_update(ddn, belief_state, action, observation, n_samples)
            new_r = get_sample_coefficients(ddn, observation_tree, new_belief, n_samples)
            r += new_r
    return r


def get_sample_ratio(cr, qr):
    if qr == 0:
        r = 1
    else:
        r = cr / qr
    return r


def get_metrics_per_run(ddn, tree, n_samples, reward_samples, time, quantum=False):
    # Calculate metrics for the time-steps
    rs, stds, samples = [], [], []
    coeffs = []

    # Initialize loop
    description = "Quantum timestep" if quantum else "Classical timestep"
    true_belief = ddn.get_belief_state()
    tbar = tqdm(range(time), total=time, desc=description, position=2, leave=False)
    for _ in tbar:
        # If run is quantum, change number of samples
        if quantum:
            cl = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples)
            coeff = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples, True)
            ratio = get_sample_ratio(cl, coeff)
            # print("ratio", ratio)
            # print("\n1", n_samples, "\n")
            n_samples_ = int(np.round(ratio * n_samples))
            # print("\n2", n_samples_, "\n")
        else:
            coeff = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples)
            n_samples_ = n_samples

        # Calculate results
        actions = pomdp_lookahead(ddn, tree, n_samples_)
        avg, cur_std = get_avg_reward_and_std(ddn, ("R", 1), {**actions, **true_belief}, reward_samples)
        
        # Belief update
        observations = ddn.sample_observation(actions)
        ddn.belief_update(actions, observations, n_samples_)
        true_belief = belief_update(ddn, true_belief, actions, observations, reward_samples)

        # Append results
        rs.append(avg)
        stds.append(cur_std)
        samples.append(n_samples_)
        coeffs.append(coeff)
        
    rs, stds, samples, coeffs = np.array(rs), np.array(stds), np.array(samples), np.array(coeffs)
    
    return rs, stds, samples, coeffs


def get_metrics(ddn, qddn, tree, config, num_runs, time):
    # Calculate metrics per run
    r = []
    
    # Get config parameters
    problem_name = config["experiment"]
    horizon = config["horizon"]
    classical_samples = config["c_sample"]
    reward_samples = config["r_sample"]
    
    # Iterate all runs
    run_bar = tqdm(range(num_runs), total=num_runs, desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, base_samples=classical_samples)
    for run_num in run_bar:
        # Get metrics for specific run
        rs, stds, _, crs = get_metrics_per_run(ddn, tree, classical_samples, reward_samples, time)
        q_rs, q_stds, q_samples, qrs = get_metrics_per_run(qddn, tree, classical_samples, reward_samples, time, True)
        
        # Append to resulting list of dicts
        runs = np.repeat(run_num, time)
        ts = np.arange(time)
        run_dict = [{
            "run": run, 
            "t": t, 
            "r": r, 
            "std": std, 
            "q_r": q_r, 
            "q_std": q_std, 
            "q_sample": q_sample,
            "c_l": c_l,
            "q_l": q_l
        } for (run, t, r, std, q_r, q_std, q_sample, c_l, q_l) in zip(runs, ts, rs, stds, q_rs, q_stds, q_samples, crs, qrs)]
        r += run_dict
    
    return r


def run_config(config, num_runs, time, save = True):
    # Extract data from config
    name = config["experiment"]
    discount = config["discount"]
    horizon = config["horizon"]
    
    # Get the ddn
    if name == "tiger":
        ddn = get_tiger_ddn(BN, discount)
        qddn = get_tiger_ddn(QBN, discount)
    elif name == "robot":
        ddn = get_robot_ddn(BN, discount)
        qddn = get_robot_ddn(QBN, discount)
    elif name == "gridworld":
        ddn = get_gridworld_ddn(BN, discount)
        qddn = get_gridworld_ddn(QBN, discount)
    
    # Build the lookahead tree
    tree = get_tree(ddn, horizon)
    
    # Get metrics
    run_dict = get_metrics(ddn, qddn, tree, config, num_runs, time)
    
    # Transform configs into dictionary for dataframe
    df = pd.DataFrame([{**config, **run_d} for run_d in run_dict])
        
    if save:
        timestamp = datetime.now(pytz.timezone('Japan')).strftime("%d_%m_%Y_%H_%M")
        filename = f"results_{timestamp}.csv"

        # Save to datasets folder if it exists, otherwise to current folder.
        if os.path.exists('datasets') and os.path.isdir('datasets'):
            filepath = os.path.join('datasets', filename)
            print("> Saved file to datasets folder.")
        else:
            filepath = filename
            print("> Saved file.")

        df.to_csv(filepath, index=False)
    return df

def run_config_wrapper(args):
    config, num_runs, tmax = args
    return run_config(config, num_runs, tmax)

def iterate_configs(configs, num_runs, tmax):
    # Create list of dictionaries as product of dictionary of lists
    configs = list(product_dict(configs))
    args = [(config, num_runs, tmax) for config in configs]

    # Create iterator function
    #def foo(config):
    #    return run_config(config, num_runs, tmax)

    ti = time.time()

    # Extract results from multiple runs in parallel
    mw = 4


    with ProcessPoolExecutor(max_workers=mw) as executor:
        try:
            # Iterate each config
            _ = list(tqdm(executor.map(run_config_wrapper, args),
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