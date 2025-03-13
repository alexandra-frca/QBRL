import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pprint import pprint

def plot_results(df, conds, unc = "std", silent = True):
    df['reward_diff'] = df['q_r'] - df['r']
    print(df.head(10))

    # Group runs with the same problem variables. 
    group_cols = ['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']

    results = {}
    for key, group in df.groupby(group_cols):
        print("key", key)

        key = dict_to_str(key, group_cols)

        ts = group['t'].unique()
        reward_diffs_by_time = {}

        # Collect reward differences for each time t
        for t, sub_group in group.groupby('t'):
            reward_diffs_by_time[t] = sub_group['reward_diff'].tolist()  # List of q_r - r values at time t

        # Convert to lists in order of time_series
        reward_diffs_list = [reward_diffs_by_time[t] for t in ts]

        # Store the result
        cr = accumulate(reward_diffs_list)

        print("Number of runs: ", len(cr[0]))
        print("> Cumulative reward differences: ")
        for i,l in enumerate(cr): 
            print(f"t={i} (average = {round(np.mean(l), 2)}): ", list(round(x, 1) for x in l))
        avgs, stds = get_avgs_stds(cr)
        print("_______________")
        print(avgs)
        print("_______________")
        results[key] = (ts, avgs, stds)
        # results[key_str] = (time_series, reward_diffs_list)

    plot(results, conds)

    if not silent:
        for key, (times, avgs, stds) in results.items():
            print(key)
            print(f"{key}")  # Readable group name
            print(f"Times: {times}")
            print(f"Average reward Differences by Time: {avgs}")
            print()

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
        # Convert np.int64 and np.float64 to Python int and float
        if isinstance(v, np.int64):
            l2.append(int(v))
        elif isinstance(v, np.float64):
            l2.append(float(v))
        else:
            l2.append(v)
        # You can add more type checks if necessary (e.g. for np.float32, np.int32, etc.)
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

def get_avgs_stds(list_of_lists):
    data = np.array(list_of_lists)
    avgs = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    return avgs, stds

def plot(results, conds):
    plt.figure(figsize=(10, 6))

    first = True
    for key, (times, avgs, stds) in results.items():
        import ast
        d = ast.literal_eval(key[6:])
        if all(d[key] == conds[key] for key in conds):
            if first:
                print("> Plotting: ")
                pprint(d)
                first = False

            label = f"c_sample = {d["c_sample"]}" 
            plt.plot(times, avgs, label=label)
            plt.fill_between(
                times,
                avgs - stds,
                avgs + stds,
                alpha=0.2
            )

    # Formatting the plot
    plt.xlabel('Time (t)')
    plt.ylabel('Cumulative Reward Difference')
    plt.title('Cumulative Reward Difference Over Time')
    plt.legend(loc='best', fontsize='small')


    plt.show()

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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.DataFrame(testdata)
df = pd.read_csv('results_oldt.csv')
df = df.iloc[4:]
df = pd.read_csv('resultsg.csv')
# df = pd.read_csv('datag.csv')
# df = df.head(200)
#df = pd.read_csv('results1103.csv')
df = pd.read_csv('results1303.csv')
df = df.iloc[20:]
print(df)
input()
# print(df.head(10))

print(df)
'''
df = df[
    (df['experiment'] == 'tiger') &
    (df['horizon'] == 2) &
    (df['c_sample'] == 5) &
    (df['t'] < 10) &
    (df['run'] < 3)
]
df['reward_diff'] = df['q_r'] - df['r']'''
# print(df)
conds = {'experiment': 'tiger', 'horizon': 2, "c_sample": 5}
plot_results(df, conds)