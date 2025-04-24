import numpy as np
import matplotlib.pyplot as plt


def classical_search(a):
    done = 0
    counter = 0
    while done == 0:
        done = np.random.binomial(1, a)
        counter += 1
    return counter

def classical_costs(aa, nruns):
    cs = []
    stds = []
    for a in aa:
        l = []
        for _ in range(nruns):
            cost = classical_search(a)
            l.append(cost)
        avg = np.mean(l)
        std = np.std(l)
        cs.append(avg)
        stds.append(std)

    return np.array(cs), np.array(stds)

def quantum_costs(aa, nruns):
    thetas = np.arcsin(aa**0.5)
    ms =  np.floor(np.pi/(4*thetas))

    ps = np.sin((2*ms+1)*thetas)**2
    cs, stds = classical_costs(ps, nruns)
    # Consider the cost of amplifying. 
    cs *= (2*ms+1) 
    return cs, stds 

def plot(x, ly, labels, shade = False, log = False):
    plt.figure(figsize=(10, 8))
    FONTSIZE = 20
    plt.xlabel("Baseline success probability P(e)", fontsize=FONTSIZE)
    plt.ylabel("Cost per sample", fontsize=FONTSIZE)

    for i, (y, l) in enumerate(zip(ly, labels)):
        colors = ['darkblue', 'firebrick']
        color = colors[i]
        linestyle = '' if log else '--'
        alpha = 1 #0.8 if log else 1
        plt.plot(x, y, marker='d', linestyle=linestyle, label = l, color = color, alpha = alpha, markersize=8)

    if log: 
        plt.plot(x, x**-1, label=r"$\mathcal{O}(P(e)^{-0.5})$", color = "darkblue", alpha = 0.6, linestyle = 'dotted')
        plt.plot(x, x**-0.5/0.63, label=r"$\mathcal{O}(P(e)^{-1})$", color = "firebrick", alpha = 0.6, linestyle = 'dotted')

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2, alpha=0.4)

    plt.gca().invert_xaxis()
    
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend(loc='best', fontsize=FONTSIZE)
    plt.show()

def plot_cost_vs_pe(nas = 10, nruns = 1000, log = True):
    nruns = 1000
    nas = 10
    aa = np.logspace(np.log10(0.005), np.log10(0.3), nas)
    cs, stds = classical_costs(aa, nruns)
    qcs, stds = quantum_costs(aa, nruns)
    labels = ["classical", "quantum"]
    plot(aa, [cs, qcs], labels, stds, log)

if __name__ == "__main__":
    plot_cost_vs_pe(log = True)
