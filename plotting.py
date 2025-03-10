import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_results0(csv):
    df = pd.read_csv(csv)

    df['reward_diff'] = df['q_r'] - df['r']

    # Group by columns that define groups, excluding 'run', 't', 'r', 'std', 
    # 'q_r', 'q_std', 'q_l', 'c_l'
    group_cols = ['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']
    df_grouped = df.groupby(group_cols + ['t'])['reward_diff'].agg(['mean', 'std']).reset_index()

    df_grouped['cumulative_mean'] = df_grouped.groupby(group_cols)['mean'].cumsum()
    # df_grouped['cumulative_std'] = df_grouped.groupby(group_cols)['std'].cumsum()
    # df_grouped['cumulative_std'] = np.sqrt(df_grouped.groupby(group_cols)['std'].apply(lambda x: (x**2).cumsum()))


    plt.figure(figsize=(10, 6))

    for key, group in df_grouped.groupby(group_cols):
        experiment, discount, horizon, c_sample, r_sample = key
        
        label = f"c_sample = {c_sample}"  # Label for the curve
        plt.plot(group['t'], group['cumulative_mean'], label=label)
        plt.fill_between(
            group['t'],
            group['cumulative_mean'] - group['cumulative_std'],
            group['cumulative_mean'] + group['cumulative_std'],
            alpha=0.2
        )

    # Formatting the plot
    plt.xlabel('Time (t)')
    plt.ylabel('Cumulative Reward Difference')
    plt.title('Cumulative Reward Difference Over Time')
    plt.legend(loc='best', fontsize='small')


    plt.show()

def plot_results1(csv, unc = "std"):
    df = pd.read_csv(csv)

    df['reward_diff'] = df['q_r'] - df['r']

    # Group by columns that define groups, excluding 'run', 't', 'r', 'std', 
    # 'q_r', 'q_std', 'q_l', 'c_l'
    group_cols = ['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']
    if unc == "std":
        df_grouped = df.groupby(group_cols + ['t'])['reward_diff'].agg(
            ['mean', 'std']).reset_index()
    if unc == "iqr":
        df_grouped = df.groupby(group_cols + ['t'])['reward_diff'].agg(
            ['mean', lambda x: x.quantile(0.75) - x.quantile(0.25)]).reset_index()
        df_grouped.rename(columns={'<lambda_0>': 'std'}, inplace=True)
        
    df_grouped['cumulative_mean'] = df_grouped.groupby(group_cols)['mean'].cumsum()
    txt = key = get_common_info(df, df_grouped)
    print(txt)


    plt.figure(figsize=(10, 6))

    for key, group in df_grouped.groupby(group_cols):
        experiment, discount, horizon, c_sample, r_sample = key
        
        label = f"c_sample = {c_sample}"  # Label for the curve
        plt.plot(group['t'], group['cumulative_mean'], label=label)
        plt.fill_between(
            group['t'],
            group['cumulative_mean'] - group['std'],
            group['cumulative_mean'] + group['std'],
            alpha=0.2
        )

    # Formatting the plot
    plt.xlabel('Time (t)')
    plt.ylabel('Cumulative Reward Difference')
    plt.title('Cumulative Reward Difference Over Time')
    plt.legend(loc='best', fontsize='small')


    plt.show()

def plot_results2(csv, unc = "std"):
    df = pd.read_csv(csv)

    df['reward_diff'] = df['q_r'] - df['r']
    df['cumulative_reward_diff'] = df.groupby(['experiment', 'discount', 'horizon', 'c_sample', 'r_sample'])['reward_diff'].cumsum()

    group_cols = ['experiment', 'discount', 'horizon', 'c_sample', 'r_sample']
    if unc == "std":
        df_grouped = df.groupby(group_cols + ['t'])['cumulative_reward_diff'].agg(['mean', 'std']).reset_index()
    elif unc == "iqr":
        df_grouped = df.groupby(group_cols + ['t'])['cumulative_reward_diff'].agg(
            ['mean', lambda x: x.quantile(0.75) - x.quantile(0.25)]).reset_index()
        df_grouped.rename(columns={'<lambda_0>': 'iqr'}, inplace=True)
        
    df_grouped['cumulative_mean'] = df_grouped.groupby(group_cols)['mean'].cumsum()
    txt = key = get_common_info(df, df_grouped)
    print(txt)


    plt.figure(figsize=(10, 6))

    for key, group in df_grouped.groupby(group_cols):
        experiment, discount, horizon, c_sample, r_sample = key
        
        label = f"c_sample = {c_sample}"  # Label for the curve
        plt.plot(group['t'], group['cumulative_mean'], label=label)
        plt.fill_between(
            group['t'],
            group['cumulative_mean'] - group['std'],
            group['cumulative_mean'] + group['std'],
            alpha=0.2
        )

    # Formatting the plot
    plt.xlabel('Time (t)')
    plt.ylabel('Cumulative Reward Difference')
    plt.title('Cumulative Reward Difference Over Time')
    plt.legend(loc='best', fontsize='small')


    plt.show()

def plot_results(df, unc = "std"):
    # df = pd.read_csv(csv)

    group_cols = ['experiment', 'discount', 'horizon', 'c_samples', 'r_samples']

    df['reward_diff'] = df['q_r'] - df['r']  # Compute reward difference

    # Group by the specified columns and extract lists of (t, reward_diff) pairs
    lists = df.groupby(group_cols).apply(lambda g: list(zip(g['t'], g['reward_diff']))).tolist()

    results = compute_avg_std(lists)
    for avgs, stds in results:
        t = [t, avg for t, avg in avgs]
        plot(a)

    print(lists)
    print(results)

def plot(avgs, std )
    plt.figure(figsize=(10, 6))

    for key, group in df_grouped.groupby(group_cols):
        experiment, discount, horizon, c_sample, r_sample = key
        
        label = f"c_sample = {c_sample}"  # Label for the curve
        plt.plot(group['t'], group['cumulative_mean'], label=label)
        plt.fill_between(
            group['t'],
            group['cumulative_mean'] - group['std'],
            group['cumulative_mean'] + group['std'],
            alpha=0.2
        )

    # Formatting the plot
    plt.xlabel('Time (t)')
    plt.ylabel('Cumulative Reward Difference')
    plt.title('Cumulative Reward Difference Over Time')
    plt.legend(loc='best', fontsize='small')


    plt.show()

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
    textstr += f"\nsamples={c_sample}"
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

testdata = {
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

df = pd.DataFrame(testdata)
df = pd.read_csv('data.csv')
print(df.head(1))
plot_results(df)