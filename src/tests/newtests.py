# -*- coding: utf-8 -*-
"""
Testing alterations to the code. 
"""
from copy import deepcopy

from src.networks.bn import BayesianNetwork
from src.networks.qbn import QuantumBayesianNetwork
from src.networks.nodes import DiscreteNode
from src.utils import RelativeTimer

# Testing the queries to a simple Bayesian network and comparing the speed of 
# the old implementation with the new. 

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
        edges.extend([#("Cloudy", "Sprinkler"), 
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

def init_problem_bn(old = False, quantum = False, simpler = False):
    specs = sprinkler_example(old = old, simpler = simpler)
    bn = create_bn(*specs, quantum = quantum, old = old)
    bn.initialize()
    return bn

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

    print(q)

def run_timed(rt, old, qiskit, Nruns = 1, Nsamples = 10):
    bn = init_problem_bn(old, quantum = True)
    rt.new()
    for i in range(Nruns):
        # q = bn.query(query=["Cloudy", "Rain", "Sprinkler", "WetGrass"], evidence={}, n_samples=1000)
        q = bn.qquery(query=["WetGrass"], 
                       evidence={"Sprinkler": 0, "Rain": 1, "Cloudy":0}, 
                       n_samples=Nsamples, qiskit = qiskit)
        # print(q)
    rt.stop()
    
def test_speeds(Nruns, Nsamples): 
    rt = RelativeTimer()
    print("> Testing speed for quantum Bayesian network queries.")
    tests = ["Old", "New", "New (no Qiskit)"]
    for s in tests:
        print(f"* {s} code")
        old = True if s=="Old" else False
        qiskit = False if s=="New (no Qiskit)" else True
        run_timed(rt, old, qiskit, Nruns, Nsamples)

# Joint probability calculations (for calculating P(e), the amplitude).

def joint_prob(bn, d):
    P = 1
    for node in list(d.keys()):
        P *= cond_prob(bn, node, d)  
    return P
        
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

def test_joint_prob(ev):
    bn = init_problem_bn(old = False, quantum = True, simpler = False)
    # ev = {"Rain": 0, "Sprinkler": 0}
    # ev = {"Rain": 0, "Cloudy": 0, "Sprinkler": 0, "WetGrass": 0}
    # ev = {"WetGrass": 0, "Cloudy": 1}
    # ev = {"Cloudy": 1, "Rain": 1, "WetGrass": 0}
    # print(bn.get_pt("Rain"))
    jp = joint_prob(bn, ev)
    print("Calculated joint probability:", jp)

def test_joint_prob2():
    # Using the method of bn class (which was tested here for convenience).
    bn = init_problem_bn()
    # Should be: 0.5*0.8*0.01 = 0.04
    jp = bn.joint_prob({"Cloudy": 1, "Rain": 1, "WetGrass": 0})
    print(jp)
