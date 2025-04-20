import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    print(cn)
    cn = df["c_l_cumsum"]*cs
    print("*", cn)
    cqn = df["q_l_cumsum"]*cs
    return cr, cqr, cn, cqn

def accumulate(df):
    df["r_cumsum"] = df.groupby("run")["r"].cumsum()
    df["q_r_cumsum"] = df.groupby("run")["q_r"].cumsum()
    df["c_l_cumsum"] = df.groupby("run")["c_l"].cumsum()
    df["q_l_cumsum"] = df.groupby("run")["q_l"].cumsum()

def plot_exectime_vs_reward(df, range, plot_diff, nbins):
    cr, cqr, cn, cqn = cumulative_qtts(df)

    # Run without range to see the individual ranges and how they overlap.
    # This info can be used to choose the common range.  
    xs, ys = bin_and_average(cr, cn, nbins, range)
    qxs, qys = bin_and_average(cqr, cqn, nbins, range)
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.plot(qxs, qys, marker='o', linestyle='-', color = "red")

    if plot_diff:
        avgxs = [(x+qx)/2 for x,qx in zip(xs, qxs)]
        diffs = [qy-y for qy,y in zip(qys, ys)]
        plt.plot(avgxs, diffs, marker='o', linestyle='-', color = "green")

    plt.show()

if __name__ == "__main__":
    # Assume experimental configs are all the same. 

    df = pd.read_csv('datasets/tiger_t=50_cs=5_nruns=100.csv')
    use_range = False 
    plot_diff = True
    nbins = 50
    if use_range:
        range = (0, 10) 
    else:
        range = None 
    plot_exectime_vs_reward(df, range, plot_diff, nbins)

