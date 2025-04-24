import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils import dfs_from_folder, filename_to_title

def sort_by_first(l1, l2):
    '''
    Gets 2 lists and sorts the elements according to the values of list 1. 

    e.g. l1 = [1, 3, 2], l2 = [A, B, C] gives [1, 2,3] [A, C, B]
    '''
    zipped = list(zip(l1, l2))
    zipped.sort()
    l1, l2 = zip(*zipped)

    l1 = list(l1)
    l2 = list(l2)

    return l1, l2

def bin_and_average(x, y, nbins, range = None):
    if range is None:
        bins = np.linspace(x.min(), x.max(), nbins + 1)
    else: 
        mn, mx = range
        ii = [i for i,k in enumerate(x) if k>=mn and k<=mx]
        x = [x[i] for i in ii]
        y = [y[i] for i in ii]
        bins = np.linspace(mn, mx, nbins + 1)

    bin_indices = np.digitize(x, bins)

    binned_df = pd.DataFrame({
        "x": x,
        "y": y,
        "bin": bin_indices
    })

    binned_means = binned_df.groupby("bin").agg({"x": "mean", "y": "mean"}).dropna()

    return binned_means["x"], binned_means["y"]

def cumulative_qtts(df):
    accumulate(df)
    
    # Assume experiment configs all same. 
    cs = df["c_sample"]

    cr = df["r_cumsum"]
    cqr = df["q_r_cumsum"]
    cn = df["c_l_cumsum"]
    cn = df["c_l_cumsum"]*cs
    cqn = df["q_l_cumsum"]*cs
    return cr, cqr, cn, cqn

def accumulate(df):
    df["r_cumsum"] = df.groupby("run")["r"].cumsum()
    df["q_r_cumsum"] = df.groupby("run")["q_r"].cumsum()
    df["c_l_cumsum"] = df.groupby("run")["c_l"].cumsum()
    df["q_l_cumsum"] = df.groupby("run")["q_l"].cumsum()

def plot_exectime_vs_reward(df, range = None, plot_diff = False, nbins = 20,
                            title = None):
    cr, cqr, cn, cqn = cumulative_qtts(df)

    # Run without range to see the individual ranges and how they overlap.
    # This info can be used to choose the common range.  
    xs, ys = bin_and_average(cr, cn, nbins, range)
    qxs, qys = bin_and_average(cqr, cqn, nbins, range)

    # Sometimes one of them gets one point too many. Want same xs. 
    n = min(len(xs), len(qxs))
    xs, ys, qxs, qys = xs[:n], ys[:n], qxs[:n], qys[:n]

    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, marker='o', linestyle='--', color = 'darkblue', label = "classical")
    plt.plot(qxs, qys, marker='o', linestyle='--', color = "firebrick", label = "quantum")

    if plot_diff:
        avgxs = [(x+qx)/2 for x,qx in zip(xs, qxs)]
        diffs = [qy-y for qy,y in zip(qys, ys)]
        plt.plot(avgxs, diffs, marker='o', linestyle='-', color = "green")

    FONTSIZE = 20
    plt.xlabel('Cumulative reward', fontsize=FONTSIZE)
    plt.ylabel(f'Cumulative cost', fontsize=FONTSIZE)
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2, alpha=0.4)
    plt.legend(loc='best', fontsize=FONTSIZE)
    if title is not None:
        plt.title(title, fontsize=FONTSIZE)
    plt.show()


def cost_vs_r_plots_from_folder(folder):
    dfs, names = dfs_from_folder(folder, return_names = True)
    for df, name in zip(dfs, names):
        plot_exectime_vs_reward(df, title = filename_to_title(name))

if __name__ == "__main__":
    # Assume experimental configs are all the same. 

    df = pd.read_csv('datasets/tiger_t=50_cs=5_nruns=100.csv')
    df = pd.read_csv('datasets/robot_t=50_cs=50_nruns=100.csv')
    use_range = False 
    plot_diff = False
    nbins = 10
    if use_range:
        range = (0, 20) 
    else:
        range = None 
        # Diff makes no sense without range because different points. 
        plot_diff = False
    #plot_exectime_vs_reward(df, range, plot_diff, nbins)
    cost_vs_r_plots_from_folder('datasets')

