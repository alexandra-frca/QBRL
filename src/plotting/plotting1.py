import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import ast
from pprint import pprint
from src.utils import dfs_from_folder, read_datasets_from_files

def plot_diff_evol_from_dfs(dfs, which_diff, conds = None, unc = "std"):
    assert which_diff in ["reward", "cost"]
    datasets = [process_data(df, which_diff) for df in dfs]
    plot_diff_evol(datasets, which_diff, conds)

def process_data(df, which_diff):
    if which_diff == "reward":
        df[which_diff] = df['q_r'] - df['r']
    elif which_diff == "cost":
        df[which_diff] = (df['q_l']/df['c_l'] - 1)#*df['c_sample']

    # Group runs with the same problem variables. 
    group_cols = ['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']

    newdatasets = {}
    for key, group in df.groupby(group_cols):
        key = dict_to_str(key, group_cols)

        ts = group['t'].unique()
        diffs_by_time = {}

        # Collect reward differences for each time t
        for t, sub_group in group.groupby('t'):
            diffs_by_time[t] = sub_group[which_diff].tolist()  
            
        diffs_list = [diffs_by_time[t] for t in ts]

        cr = accumulate(diffs_list)
        
        avgs, rg = get_avgs_shade(cr)
        newdatasets[key] = (ts, avgs, rg) 

    return newdatasets

def plot_diff_evol(datasets, which_diff, conds = None, trange = "standard"):
    # trange: whether to fit the xaxis range to smallest/biggest among datasets.
    # assert trange in ["smallest", "largest"]
    plt.figure(figsize=(10, 8))
    lts = []
    for i, dataset in enumerate(datasets):
        colors = ['tomato', 'seagreen'] 
        color = colors[i]
        times = plot_dataset(dataset, conds, color, which_diff)
        lts.append(times)

    mintimes = [min(ts) for ts in lts]
    maxtimes = [max(ts) for ts in lts]

    # Plot common range.
    if trange == "smallest":
        xmin, xmax = max(mintimes), min(maxtimes)
    #else:
    #    xmin, xmax = min(mintimes), max(maxtimes)
        plt.xlim(xmin, xmax)
        adjust_ylim(xmin, xmax)

    FONTSIZE = 20
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel(f'Cumulative {which_diff} difference', fontsize=FONTSIZE)
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2, alpha=0.4)

    if len(datasets) > 1:
        plt.legend(loc='best', fontsize=FONTSIZE)

    plt.show()

def adjust_ylim(xmin, xmax):
    ys_keep = []
    for line in plt.gca().get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for x, y in zip(xdata, ydata):
            if xmin <= x <= xmax:
                ys_keep.append(y)

    from matplotlib.collections import PolyCollection
    for patch in plt.gca().collections:
        if isinstance(patch, PolyCollection):
            for path in patch.get_paths():
                vertices = path.vertices
                xdata = vertices[:, 0]
                ydata = vertices[:, 1]
                for x, y in zip(xdata, ydata):
                    if xmin <= x <= xmax:
                        ys_keep.append(y)

    if ys_keep:
        ymin = min(ys_keep)
        ymax = max(ys_keep)
        plt.ylim(ymin, ymax)

def plot_dataset(dataset, conds, color, which_diff):
    for key, (times, avgs, rg) in dataset.items():
        d = ast.literal_eval(key[6:])
        if conds is None or all(d[key] == conds[key] for key in conds):
            label = f"{d['experiment']} problem"
            if which_diff == "reward":
                label += f", c_samples = {d['c_sample']}" 

            plt.plot(times, avgs, label=label, color = color, linestyle='--',
                     linewidth=1, marker='o', markersize=5)
            plt.fill_between(
                times,
                rg[0], #[rg[0] for rg in rgs],
                rg[1], # [rg[1] for rg in rgs],
                color = color,
                alpha=0.1,
                edgecolor=None
            )
    return times

def dict_to_str(key, names):
    '''
    Turn dictionary into a string.
    '''
    key = convert_numpy_types(key)
    key_dict = {col: val for col, val in zip(names, key)}
    key = f"Group {key_dict}"
    return key

def convert_numpy_types(l):
    l2 = []
    for v in l:
        if isinstance(v, np.int64):
            l2.append(int(v))
        elif isinstance(v, np.float64):
            l2.append(float(v))
        else:
            l2.append(v)
    return l2

def accumulate(data):
    '''
    for instance 
    [[1, 2, 3], [2, 10, 20]]
    becomes 
    [[1, 2, 3], [3, 12, 23]]
    '''
    data_array = np.array(data)
    csums = np.cumsum(data_array, axis=0)
    csums = csums.tolist()
    return csums

def get_avgs_shade(list_of_lists, shaded = "std"):
    assert shaded in ["std", "iqr"]
    data = np.array(list_of_lists)
    avgs = np.mean(data, axis=1)
    if shaded == "std":
        std = np.std(data, axis=1)
        rg = avgs - std, avgs + std
    elif shaded == "iqr":
        rg = np.percentile(data, 25, axis=1), np.percentile(data, 75, axis=1) 
    return avgs, rg

def get_csample_from_dictstr(dstr):
    import re
    pattern = r"\'c_sample\':\s*(\d+)"
    match = re.search(pattern, dstr)

    assert match
    c_sample = int(match.group(1))  
    return c_sample

def compute_avg_std(grouped_lists):
    results = []

    for lst in grouped_lists:
        # Dictionary to collect values by t
        values_by_t = defaultdict(list)

        # Populate dictionary with t values
        for t, reward_diff in lst:
            values_by_t[t].append(reward_diff)

        # Compute mean and std for each unique t
        avg_list = sorted((t, np.mean(values)) for t, values in values_by_t.items())
        std_list = sorted((t, np.std(values, ddof=0)) for t, values in values_by_t.items())  # ddof=0 for population std

        results.append((avg_list, std_list))

    return results


def get_common_info(df, df_grouped):
    for key, group in df_grouped.groupby(['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']):
        experiment, discount, horizon, c_sample, r_sample = key
    
    # Compute number of unique runs for this group
    nruns = df[(df['experiment'] == experiment) & 
               (df['discount'] == discount) & 
               (df['horizon'] == horizon) & 
               (df['c_sample'] == c_sample) & 
               (df['r_sample'] == r_sample)]['run'].nunique()
    
    textstr = f"$H = {horizon}"
    textstr += f"\ngamma={discount}"
    textstr += f"\nsample={c_sample}"
    textstr += f"\nruns={nruns}"

    return textstr

testdata = {
    'experiment': ['A', 'A', 'A', 'B', 'B', 'B'],
    'discount': [0.9, 0.9, 0.9, 0.95, 0.95, 0.95],
    'horizon': [100, 100, 100, 200, 200, 200],
    'c_sample': [10, 10, 10, 20, 20, 20],
    'r_sample': [5, 5, 5, 10, 10, 10],
    'run': [1, 1, 1, 2, 2, 2],
    't': [0, 1, 2, 0, 1, 2],
    'r': [0.5, 0.6, 0.4, 0.7, 0.8, 0.9],
    'q_r': [0.7, 0.8, 0.5, 0.8, 0.9, 1.0],
}

testdata2 = {
    'experiment': ['A', 'A', 'A', 'A', 'A', 'A'],
    'discount': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'horizon': [100, 100, 100, 100, 100, 100],
    'c_sample': [10, 10, 10, 10, 10, 10],
    'r_sample': [5, 5, 5, 5, 5, 5],
    'run': [1, 1, 1, 2, 2, 2],
    't': [0, 1, 2, 0, 1, 2],
    'r': [0.5, 0.6, 0.4, 0.7, 0.8, 0.9],
    'q_r': [0.7, 0.8, 0.5, 0.8, 0.9, 1.0],
}

def diff_plots_from_folder(folder):
    dfs = dfs_from_folder(folder)
    for which_diff in ['reward', 'cost']:
        plot_diff_evol_from_dfs(dfs, which_diff)

if __name__ == "__main__":
    filenames = ["robot_t=20_cs=50_nruns=90.csv"]#,
                 #'tiger_t=50_cs=5_nruns=100.csv']
    
    # conds = {'experiment': 'tiger', 'horizon': 2, "c_sample": 5}

    all_datasets = True
    which = 0
    which_diff = 'reward' if which == 0 else 'cost'
    folder = 'datasets'
    if all_datasets: 
        dfs = dfs_from_folder(folder)
    else:
        dfs = read_datasets_from_files(filenames)
    plot_diff_evol_from_dfs(dfs, which_diff)
    
    