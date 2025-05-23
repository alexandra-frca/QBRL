from __future__ import annotations
from itertools import product
from typing import Union
import pandas as pd
import numpy as np
import os

# Types
Id = tuple[str, int]
Value = Union[int, float]
BeliefState = dict[Id, pd.DataFrame]
DDN = "DynamicDecisionNetwork"


def is_bit_value(number: int, index: int, value: int) -> bool:
    """Checks if bit number `index` of binary representation of `number`
    has value `value`
    """
    is_set = bool((number >> index) & 1)
    return is_set if bool(value) else not is_set


def are_bit_values(number: int, value_dict: dict[int, int]) -> bool:
    return all([is_bit_value(number, k, v) for k, v in value_dict.items()])


def product_dict(my_dict: dict):
    keys = my_dict.keys()
    values = my_dict.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))
        
        
def get_string_elems(s: str, indices: list[int]) -> str:
    r = ""
    for i in indices:
        r += s[i]
    return r

        
def df_binary_str_filter(df: pd.DataFrame, col: str, bin_dict: dict[int, int], value_space: list[Value]) -> pd.DataFrame:
    mask = lambda i: are_bit_values(value_space.index(df[col].iloc[i]), bin_dict)
    indices = [i for i in range(len(df)) if mask(i)]
    return df.iloc[indices]
        

def df_dict_filter(df: pd.DataFrame, dict_filter: dict):
    return df.loc[df[dict_filter.keys()].isin(dict_filter.values()).all(axis = 1),: ]
        
        
def belief_update(ddn: DDN, belief_state: BeliefState, actions: dict[Id, Value], observations: dict[Id, Value], n_samples: int = 100, epsilon: float = 1e-3) -> dict[Id, pd.DataFrame]:
        # TODO: check if actions and observations dict are correct
        
        # Get next state nodes
        node_filter = lambda _, n: (n.get_type() == ddn.state_type) and (n.get_time() == ddn.get_time() + 1)
        query = ddn.get_nodes_by_key(node_filter)
        
        # Query the next belief-state
        evidence = {**belief_state, **actions, **observations}
        next_belief = ddn.query(query, evidence, n_samples, print_pe = False)
        
        # TODO: Check if there is a better way to do this
        # Reduce column time
        col_replace = {(n, t): (n, t-1) for (n, t) in query}
        next_belief.rename(columns=col_replace, inplace=True)
        
        # Change CPT for each root state node
        r = {}
        for nid in query:
            # Get columns to keep in node CPT
            nid = col_replace[nid]
            nid_query = list(ddn.get_node(nid).get_pt().columns)
            
            # Construct CPT by summing over other variables
            df = next_belief.groupby(nid_query).sum().reset_index()
            
            # Remove zero-entries with epsilon
            # df[df["Prob"] == 0] = epsilon
            df["Prob"] /= df["Prob"].sum()
            r[nid] = df
                
        return r


def get_rewards_and_weights(ddn: DDN, reward_node: Id, evidence: dict[Id, Value], n_samples) -> Value:
    reward_df = ddn.query([reward_node], evidence, n_samples)
    rewards, weights = reward_df[reward_node].to_numpy(), reward_df["Prob"].to_numpy()
    return rewards, weights


def get_avg_reward_and_std(ddn: DDN, reward_node: Id, evidence: dict[Id, Value], n_samples) -> Value:
    rewards, weights = get_rewards_and_weights(ddn, reward_node, evidence, n_samples)
    avg = np.average(rewards, weights=weights)
    std = np.sqrt(np.average((avg - rewards)**2, weights=weights))
    return avg, std

from time import perf_counter

class RelativeTimer():
    def __init__(self):
        self.ti = perf_counter()
        self.reference = None
        
    def new(self):
        self.ti = perf_counter()
        
    def stop(self):
        tf = perf_counter()
        Dt = tf - self.ti 
        
        if self.reference is None:
            self.reference = Dt
            print(f"> Time elapsed: {Dt*1e-3}ms.")
            print("  (To be used as reference.)")
        else:
            print(f"> Time elapsed: {Dt*1e-3}ms.")
            print(f"  ({round(self.reference/Dt,1)}x faster than the reference.)")
        return Dt
    
def dfs_from_folder(folder, return_names = False):
    filenames = os.listdir(folder)
    dfs = []
    for file in filenames: 
        df = pd.read_csv(folder + '/' + file)
        dfs.append(df)
    if return_names: 
        return dfs, filenames
    return dfs

def read_datasets_from_files(filenames, folder = "datasets/"):
    dfs = []
    for file in filenames: 
        df = pd.read_csv(folder + file)
        dfs.append(df)
    return dfs

def filename_to_title(s):
    '''
    E.g. 'robot_t=50_cs=50_nruns=100.csv' -> 'Robot problem (t=50, cs = 50, nruns=100)
    '''
    s = s[0].upper() + s[1:]
    s = s.replace('_', ' problem (', 1)
    s = s.replace('_', ', ')
    s = s.replace('.csv', ')')
    return s