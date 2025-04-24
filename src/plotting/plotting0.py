import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils import dfs_from_folder, filename_to_title

def accumulate(df):
    df["r_cumsum"] = df.groupby("run")["r"].cumsum()
    df["q_r_cumsum"] = df.groupby("run")["q_r"].cumsum()

def get_avgs_shade(df, shaded = "std"):
    assert shaded in ["std", "iqr"]
    ts = df['t'].unique()
    accumulate(df)

    lts, lavgs, lrgs = [], [], []
    for col in ["r_cumsum", "q_r_cumsum"]:
        r_by_time = [group[col].tolist() for _, group in df.groupby("t")]

        avgs = np.mean(r_by_time, axis = 1) 

        if shaded == "std":
            std = np.std(r_by_time, axis=1)
            rg = avgs - std, avgs + std
        elif shaded == "iqr":
            rg = np.percentile(r_by_time, 25, axis=1), np.percentile(r_by_time, 75, axis=1) 

        lts.append(ts)
        lavgs.append(avgs)
        lrgs.append(rg)

    return lts, lavgs, lrgs

def plot_cr_evol(data, title = None):
    plt.figure(figsize=(10, 8))

    colors = ['darkblue', 'firebrick']
    labels = ['classical', 'quantum']
    for ts, avgs, rg, color, label in zip(*data, colors, labels):
        plt.plot(ts, avgs, color = color, label = label,
                linestyle='--', linewidth=1, marker='o', markersize=5)
        plt.fill_between(
            ts,
            rg[0], 
            rg[1],
            color = color,
            alpha=0.1,
            edgecolor=None
        )
    
    FONTSIZE = 20
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel(f'Cumulative reward', fontsize=FONTSIZE)
    plt.minorticks_on()
    plt.legend(loc='best', fontsize=FONTSIZE)
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2, alpha=0.4)
    if title is not None:
        plt.title(title, fontsize=FONTSIZE)
    plt.show()

def cr_plots_from_folder(folder):
    dfs, names = dfs_from_folder(folder, return_names = True)
    for df, name in zip(dfs, names):
        plottingdata = get_avgs_shade(df)
        plot_cr_evol(plottingdata, title = filename_to_title(name))

if __name__ == "__main__":
    which = 1
    if which == 0:
        df = pd.read_csv('datasets/tiger_t=50_cs=5_nruns=100.csv')
    else:
        df = pd.read_csv('datasets/robot_t=50_cs=50_nruns=100.csv')
    
    plottingdata = get_avgs_shade(df)
    plot_cr_evol(plottingdata)

