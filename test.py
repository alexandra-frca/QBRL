# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:58:36 2024

@author: alexa
"""

from src.networks.bn import BayesianNetwork
from src.networks.qbn import QuantumBayesianNetwork
from src.networks.nodes import DiscreteNode
from src.utils import RelativeTimer



def sprinkler_pts(simpler = False):
    data = {}
    data["Cloudy"] = {"Cloudy": [0,1], "Prob": [0.5,0.5]}
    data["Sprinkler"] = {"Cloudy": [0,0,1,1], "Sprinkler": [0,1,0,1], 
                         "Prob": [0.5,0.5,0.9,0.1]}
    if not simpler:
        data["Rain"] = {"Cloudy": [0,0,1,1], "Rain": [0,1,0,1], 
                        "Prob": [0.8,0.2,0.2,0.8]}
        data["WetGrass"] = {"Sprinkler": [0,0,0,0,1,1,1,1], 
                            "Rain": [0,0,1,1,0,0,1,1], 
                            "WetGrass": [0,1,0,1,0,1,0,1], 
                            "Prob": [1,0,0.1,0.9,0.1,0.9,0.01,0.99]}
    else:
        data["Cloudy"] = {"Cloudy": [0,1], "Prob": [0.1,0.9]}
    return data
    
def sprinkler_example(old = False, simpler = False):
    names = ["Cloudy", "Sprinkler"] 
    if not simpler:
        names.extend(["Rain", "WetGrass"])
        
    nodes = [DiscreteNode(name, "state", value_space=[0, 1], old = old) 
             for name in names]
    
    edges = [("Cloudy", "Sprinkler")]
    if not simpler:
        edges.extend([("Cloudy", "Sprinkler"), 
                      ("Cloudy", "Rain"), 
                      ("Sprinkler", "WetGrass"), 
                      ("Rain", "WetGrass")])
    
    data = sprinkler_pts(simpler)
    return nodes, edges, data

def create_bn(nodes, edges, data, quantum, old = False):
    if quantum:
        bn = QuantumBayesianNetwork(old)
    else:
        bn = BayesianNetwork(old)

    bn.add_nodes(nodes)
    bn.add_edges(edges)
    for name in data.keys():
        bn.add_pt(name, data[name])
    return bn

'''
specs = sprinkler_example()
bn = create_bn(*specs, quantum = False)
bn.initialize()
bn.query(query=["Cloudy", "WetGrass"], evidence={"Cloudy": 1}, n_samples=1)
import pandas as pd 

df = pd.DataFrame({'A': [0, 1, 1],
                   'B': [1, 0, 0],
                   'C': [1, 1, 0],
                   'Prob': [0.3, 0.4, 0.3]})
print(df)
print("...........")
df = df.groupby(['A', 'B']).sum()
print(df)


'''

def init_problem_bn(old = False, quantum = False, simpler = False):
    specs = sprinkler_example(old = old, simpler = simpler)
    bn = create_bn(*specs, quantum = quantum, old = old)
    bn.initialize()
    return bn

def test_joint_prob():
    bn = init_problem_bn()
    # Should be: 0.5*0.8*0.01 = 0.04
    jp = bn.joint_prob({"Cloudy": 1, "Rain": 1, "WetGrass": 0})
    print(jp)

def test_qquery(simpler = False):
    if simpler:
        bn = init_problem_bn(old = False, quantum = True, simpler = True)
        ev = {"Cloudy": 0}
        q = bn.qquery(query=["Sprinkler"], 
                     evidence=ev, 
                     n_samples=50)
    else:
        bn = init_problem_bn(old = False, quantum = True, simpler = False)
        ev = {"Rain": 0}
        ev = {"Cloudy": 0, "Sprinkler": 0}
        # ev = {"Rain": 0, "Cloudy": 0, "Sprinkler": 0}
        q = bn.qquery(query=["Sprinkler"], 
                     evidence=ev, 
                     n_samples=50)

    # print(q)

def run_timed(rt, old, quantum = True, N = 1):
    bn = init_problem_bn(old, quantum)
    rt.new()
    for i in range(N):
        # q = bn.query(query=["Cloudy", "Rain", "Sprinkler", "WetGrass"], evidence={}, n_samples=1000)
        q = bn.qquery(query=["WetGrass"], 
                       evidence={"Sprinkler": 0, "Rain": 1, "Cloudy":0}, 
                       n_samples=200)
        # print(q)
    rt.stop()
    
def test_speeds(): 
    rt = RelativeTimer()
    for old in [True, False]:
        run_timed(rt, old)

# test_qquery()
# run_timed(rt, False, quantum = True)
test_speeds()

from copy import deepcopy

def joint_prob(bn, d):
    P = 1
    for node in list(d.keys()):
        P *= cond_prob(bn, node, d)  
    return P
        
dicts = []
def cond_prob(bn, node, d):
    # print(node, d)
    val = d[node]
    ids = list(d.keys())
    vals = [d[var] for var in ids]
    parents = bn.get_parents(node)
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
        cp = cond_prob_aux(bn, node, val, ps, pvals)
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
    cP = df.loc[(df[lnodes] == lvals).all(axis = 1), 'Prob']
    cP = cP.iat[0]
    return cP

def test_joint_prob2():
    bn = init_problem_bn(old = False, quantum = True, simpler = False)
    ev = {"Rain": 0, "Sprinkler": 0}
    # ev = {"Rain": 0, "Cloudy": 0, "Sprinkler": 0, "WetGrass": 0}
    ev = {"WetGrass": 0, "Cloudy": 1}
    # print(bn.get_pt("Rain"))
    jp = joint_prob(bn, ev)
    print(jp)


# test_joint_prob2()




