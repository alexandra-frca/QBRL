import pandas as pd 
from get_ddns import get_tiger_ddn
from src.networks.bn import BayesianNetwork as BN

def joint_prob(bn, d):
    P = 1
    for node in list(d.keys()):
        # print("Node", node)
        P *= cond_prob(bn, node, d)  
        # print("P", P)
    return P
        
def cond_prob(bn, node, d):
    
    # print(node, d)
    val = d[node]
    ids = list(d.keys())
    vals = [d[var] for var in ids]
    parents = bn.get_parents(node)
    # print("fun parents", parents)
    # Topological order.
    parents = [node for node in bn.node_queue if node in parents]
    
    # If a parent with higher topological order is specified, no need for 
    # other. Can find path to calculate probability without it.
    # Remember no loops. If nodes with same order, index is arbitrary
    absent_parents = [parent for parent in parents if parent not in d]
    present_parents = [parent for parent in parents if parent in d]
    for ap in absent_parents:
        for parent in present_parents:
            if parents.index(ap) > parents.index(parent):       
                absent_parents.remove(ap)
            
    if not absent_parents:
        # All parent values specified.
        pvs = [(p,v) for (p,v) in zip(ids, vals) if p in parents]
        if pvs:
            # Split into list of parent ids, list of parent values.
            ps, pvals = map(list, zip(*pvs))
        else:
          ps, pvals = [], []
        cp = bn.cond_prob_aux(node, val, ps, pvals)
        return cp
    else:
        # Need recursivity to consider possible parent values.
        parent = parents[0]
        print(parent)
        vals = [0, 1]
        
        ds = [deepcopy(d) for val in vals]
        for d, val in zip(ds,vals):
            d[parent] = val
            
        ps = [cond_prob(bn, parent, d) for d in ds]
        # print(d)
        
        return sum([p*cond_prob(bn, node, d) for p,d in zip(ps, ds)])

def cond_prob_aux(bn, node, nval, parents, pvals):
    # P(node|parents) where all parent values specified. Works for no parents.
    df = bn.get_pt(node)
    lnodes = parents + [node]
    lvals = pvals + [nval]
    # print("lnodes", lnodes)
    # print("lvals", lvals)
    cP = df.loc[(df[lnodes] == lvals).all(axis = 1), 'Prob']
    cP = cP.iat[0]
    return cP

which = 0

if which==0:
    discount = 0.9
    ddn = get_tiger_ddn(BN, discount)
    v0 = 0
    v1 = pd.DataFrame({('S', 0): [0, 1], 'Prob': [0.5, 0.5]})
    # ev = {("A", 0): v0, ("S",0): v1}
    ev = {("A", 0): v0, ("O",1): 0}
else:
    from src.new.newtests import init_problem_bn
    ddn = init_problem_bn(old = False, quantum = True, simpler = False)
    ev = {"Cloudy": 1, "Rain": 1, "WetGrass": 0}

import matplotlib.pyplot as plt
# ddn.draw()
# plt.show()
backup_pts = ddn.encode_evidence(ev)
# self.decode_evidence(backup_pts)
print("****** fun ********")
jp = joint_prob(ddn, ev)
print("\n Joint probability: ", jp)
print("****** class ********")
jp = ddn.joint_prob(ev)
print("\n Joint probability: ", jp)
# joint_prob(ddn, d)