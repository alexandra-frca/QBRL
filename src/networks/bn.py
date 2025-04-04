from __future__ import annotations
from src.networks.nodes import DiscreteNode
from typing import Callable, Union
import networkx as nx
import pandas as pd
from copy import deepcopy

# Defining types
Id = Union[str, tuple[str, int]]
Edge = tuple[Id, Id]
Value = Union[int, float]
Evidence = Union[Value, pd.DataFrame]
ProbTable = Union[dict[Id, list[Value]], pd.DataFrame]


class BayesianNetwork:
    """
    A class for a discrete random variable Bayesian network.
    It leverages the DiscreteNode class and implements rejection sampling for inference.
    """

    def __init__(self, old = False):
        # The node map maps the node's ids to the node objects themselves
        self.node_dict: dict[Id, DiscreteNode] = {}
        
        # The graph maps node ids to list of children node ids
        self.graph: dict[Id, list[Id]] = {}
        
        # The node queue lists the topological ordering of the nodes, for inference traversal
        self.node_queue: list[Id] = None
        
        self.old = old

        self.quantum = False
        
    def get_node_queue(self) -> list[Id]:
        return self.node_queue
    
    def get_node_dict(self) -> dict[Id, DiscreteNode]:
        return self.node_dict
    
    def get_graph(self) -> dict[Id, list[Id]]:
        return self.graph

    def draw(self):
        # Create a networkx directed graph
        G = nx.DiGraph(directed=True)
        
        # Convert edges into strings and add them to the graph
        edges = list(map(lambda x: (str(x[0]), str(x[1])), self.get_edges()))
        G.add_edges_from(edges)
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Define options for networkx draw method
        options = {
            'node_color': 'orange',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(G, pos, arrows=True, **options)

    def add_nodes(self, nodes: list[DiscreteNode]):
        # Iterate every node to be added
        for node in nodes:
            # Make sure it does not already exist
            if node.get_id() not in self.node_dict:
                self.node_dict[node.get_id()] = node
                self.graph[node.get_id()] = []

    def add_edges(self, edges: list[Edge]):
        # TODO: handle cases where source or destination nodes are not in the Bayesian network.
        # Iterate every edge to be added
        for s, d in edges:
            self.graph[s].append(d)

    def gen_node_queue(self) -> list[Id]:
        """Create the topological node ordering of the Bayesian network using Khan's algorithm.
        This method should only be called once the network structure has been completely defined.

        Returns:
            list[Id]: list of nodes in the Bayesian Network in topological order.
        """
        nodes = [n for n in self.node_dict if self.is_root(n)]
        
        while len(nodes) < len(self.node_dict):
            for node in self.node_dict:
                if node not in nodes:
                    parents = self.get_parents(node)
                    # Add node to list if all its parents are on the list (safe to traverse)
                    if set(parents).issubset(nodes):
                        nodes.append(node)
        return nodes

    def initialize(self):
        self.node_queue = self.gen_node_queue()
        
    def get_node(self, nid: Id):
        return self.node_dict[nid]

    def get_nodes(self) -> list[Id]:
        return self.node_dict.keys()

    def get_edges(self) -> list[Edge]:
        return [(s, d) for s in self.graph for d in self.graph[s]]

    def get_parents(self, node_id: Id) -> list[Id]:
        return [k for k in self.graph if node_id in self.graph[k]]

    def get_pt(self, node_id: Id) -> pd.DataFrame:
        return self.node_dict[node_id].get_pt()
    
    def add_pt(self, node_id: Id, pt: pd.DataFrame):
        self.node_dict[node_id].add_pt(pt)
        
    def fix_value(self, node_id: Id, value: int):
        self.node_dict[node_id].fix_value(value)

    def is_leaf(self, node_id: Id) -> bool:
        return len(self.graph[node_id]) == 0

    def is_root(self, node_id: Id) -> bool:
        return len(self.get_parents(node_id)) == 0

    def add_pt(self, node_id: Id, pt: ProbTable):
        self.node_dict[node_id].add_pt(pt)

    def get_nodes_by_type(self, node_type: str) -> list[Id]:
        return [k for k, v in self.node_dict.items() if v.get_type() == node_type]
    
    def get_nodes_by_key(self, key: Callable[[Id, DiscreteNode], bool]):
        return [n for n in self.node_dict.keys() if key(n, self.node_dict[n])]
    
    def encode_evidence(self, evidence: dict[Id, int]) -> dict[Id, pd.DataFrame]:
        # Fix the value of every root node in evidence for faster inference
        root_nodes = [n for n in self.get_nodes() if (self.is_root(n) and n in evidence)]
        backup_pts = {r: self.get_pt(r) for r in root_nodes}
        for r in root_nodes:
            if isinstance(evidence[r], pd.DataFrame):
                self.add_pt(r, evidence[r])
                evidence.pop(r)
            else:
                self.fix_value(r, evidence[r])
        return backup_pts
    
    def decode_evidence(self, backup_pts: dict[Id, pd.DataFrame]):
        for r in backup_pts:
            self.add_pt(r, backup_pts[r])
    
    def get_sample(self) -> dict[Id, int]:
        """Returns a sample from every node using the direct sampling algorithm. 
        Uses the DiscreteNode class sample method.
        Should only be called after the Bayesian network has been initialized, otherwise returns empty dict.

        Returns:
            dict[Id, int]: a dictionary mapping node ids to their respective sample values.
        """
        # Create empty sample
        sample = {}

        # Sample a result from each node
        for node in self.node_queue:
            sample[node] = self.node_dict[node].get_sample(sample)
            
        return sample
    
    def query_old(self, query: list[Id], evidence: dict[Id, Evidence] = None, n_samples: int = 100) -> pd.DataFrame:
        """
        Applies the rejection sampling algorithm to approximate any probability distribution.

        Arguments:
            - query ([Id]): node ids for the random variables of the desired probability distribution.
            - evidence ({Id: int}): values for random variables as evidence for the inference. Defaults to None.
            - n_samples (int): number of samples to retrieve. Defaults to 100.

        Return (pd.Dataframe): a dataframe that represents the inferred posterior distribution.
        """
        
        # TODO: throw error when evidence DataFrame is not a root node
        
        samples = []
        evidence = {} if evidence is None else evidence
        backup_pts = self.encode_evidence(evidence) # Fix the value of every root node in evidence for faster inference

        # Create multiple samples
        num_samples = 0
        while (num_samples < n_samples):
            sample = self.get_sample() # Extract sample from the BN
            matches = [sample[name] == evidence[name] for name in evidence] # Store sample if it matches with evidence
            if all(matches):
                samples.append(sample)
                num_samples += 1
                
        self.decode_evidence(backup_pts) # Re-change probability tables of root nodes in evidence
        sample_df = pd.DataFrame(samples)
        sample_df = sample_df.value_counts(normalize=True).to_frame("Prob") # Turn result into probability table
        sample_df = sample_df.groupby(query).sum().sort_values(query).reset_index() # Group over query variables and sum over all other variables

        return sample_df

    def query(self, query: list[Id], evidence: dict[Id, Evidence] = None, 
              n_samples: int = 100, quantum = True, print_pe = False) -> pd.DataFrame:
        """
        Applies the rejection sampling algorithm to approximate any probability distribution.

        Arguments:
            - query ([Id]): node ids for the random variables of the desired probability distribution.
            - evidence ({Id: int}): values for random variables as evidence for the inference. Defaults to None.
            - n_samples (int): number of samples to retrieve. Defaults to 100.

        Return (pd.Dataframe): a dataframe that represents the inferred posterior distribution.
        """
        
        if self.old:
            return self.query_old(query, evidence, n_samples)
        if self.quantum and quantum:
            # Second flag: avoid infinite recursion in the case qquery calls
            # query because there's no need to run the quantum version. 
            # (post 15/03/25). 
            return self.qquery(query, evidence, n_samples, print_pe = print_pe)
        
        samples = []
        evidence = {} if evidence is None else evidence
        
        # Assign the specific values of evidence vars that are root nodes.
        # I.e. instead of table having 2 values (0 or 1), keep only correct one. 
        # Will be restored later.
        backup_pts = self.encode_evidence(evidence)
        
        
        # Create multiple samples
        num_samples = 0
        while (num_samples < n_samples):
            # Extract sample from the BN.
            sample = self.get_sample() 
            # Store sample if it matches with evidence.
            matches = [sample[name] == evidence[name] for name in evidence] 
            if all(matches):
                samples.append(sample)
                num_samples += 1
                
        
        self.decode_evidence(backup_pts)

        # Table with 1 sample per line.
        sample_df = pd.DataFrame(samples)
        # Join identical samples in the same line, calculate relative frequency.
        sample_df = sample_df.value_counts(normalize=True).to_frame("Prob") 
        # Join samples whose query variables are the same, sum over others.
        sample_df = sample_df.groupby(query).sum()
        # Order and make into enumerated dataframe. 
        sample_df = sample_df.sort_values(query).reset_index()

        return sample_df
        
    def joint_prob(self, d):
        # d: dictionary with the evidence
        P = 1
        print("d", d)
        for node in list(d.keys()):
            print("Node", node)
            P *= self.cond_prob(node, d)  
            print("P", P)
            '''
            if not isinstance(d[node], pd.DataFrame):
                print("Node", node)
                P *= self.cond_prob(node, d)  
                print("P", P)
            else:
                print("Ev val was dataframe")
            '''
        return P
            
    def cond_prob(self, node, d):
        val = d[node]
        ids = list(d.keys())
        vals = [d[var] for var in ids]
        parents = self.get_parents(node)
        # Topological order.
        parents = [node for node in self.node_queue if node in parents]
        
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
            
            cp = self.cond_prob_aux(node, val, ps, pvals)
            return cp
        else:
            # Need recursivity to consider possible parent values.
            # tparents = [node for node in self.node_queue if node in parents]
            parent = parents[0]
            vals = [0, 1]
            
            ds = [deepcopy(d) for val in vals]
            for d, val in zip(ds,vals):
                d[parent] = val
                
            ps = [self.cond_prob(parent, d) for d in ds]
            
            return sum([p*self.cond_prob(node, d) for p,d in zip(ps, ds)])
    
            
    
    def cond_prob_aux(self, node, nval, parents, pvals):
        # P(node|parents) where all parent values specified. Works for no parents.
        df = self.get_pt(node)
        '''
        if df is None: 
            return 1
            print("Check this!")
        '''
        lnodes = parents + [node]
        lvals = pvals + [nval]
        # print("df", df)
        # print("lnodes", lnodes)
        # print("df[lnodes]", df[lnodes])
        # print("lvals", lvals)
        cP = df.loc[(df[lnodes] == lvals).all(axis = 1), 'Prob']
        # print("cP", cP)
        cP = cP.iat[0]
        return cP
        
    '''
    def joint_prob(self, d: dict[Id, Value]):
        ids = list(d.keys())
        vals = [d[var] for var in ids]
        P = 1
        for node, val in zip(ids, vals):
            parents = self.get_parents(node)
            pvs = [(p,v) for (p,v) in zip(ids, vals) if p in parents]
            
            if pvs:
              ps, pvals = map(list, zip(*pvs))
            else:
              ps, pvals = [], []
            
            cp = self.cond_prob(node, val, ps, pvals)
            # print("* ", node, ps, cp)
            
            P *= cp
        return P
    
    def cond_prob(self, node, nval, parents, pvals):
        # P(node|parents) 
        df = self.get_pt(node)
        lnodes = parents + [node]
        lvals = pvals + [nval]
        cP = df.loc[(df[lnodes] == lvals).all(axis = 1), 'Prob']
        return cP.iat[0]
    '''