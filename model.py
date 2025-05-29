import random
import networkx as nx
import numpy as np
from mesa.model import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from agent import OpinionAgent
from utils import bounded_gamma

class OpinionNetworkModel(Model):
    """
    Opinion dynamics model on a directed social network with weighted edges.
    Supports multiple topologies and edge-weight configurations:
    - Reciprocal edges share weight in both directions.
    - One-way edges (influencer→non-influencer) have separate weight.
    """
    def __init__(
        self,
        num_agents=100,
        influencer_prob=0.1,
        self_weight_max=3.0,
        reciprocal_max=1.0,
        one_way_max=1.0,
        ba_m=2,
        ba_seed=None,
        seed=42,
        hda_influence=0.8,
        bias_exp_lambda=1.0,  # default value, will be overridden by UI if present
    ):
        super().__init__()
        self.num_agents = num_agents
        self.steps = 0  
        self.hda_influence = hda_influence
        self.bias_exp_lambda = bias_exp_lambda
        self.influencer_prob = influencer_prob
        self.self_weight_max = self_weight_max
        self.rec_max = reciprocal_max
        self.one_max = one_way_max
        self.ba_m = ba_m

        # Set random seed for reproducibility
        if seed is not None:
            try:
                seed_int = int(seed)
            except Exception:
                seed_int = None
            if seed_int is not None:
                random.seed(seed_int)
                np.random.seed(seed_int)

        # Build base undirected graph (Barabási–Albert)
        G0 = nx.barabasi_albert_graph(n=self.num_agents, m=self.ba_m, seed=seed)

        # Prepare directed graph with weights
        self.G = nx.DiGraph()
        self.G.add_nodes_from(G0.nodes())

        # Assign influencer flags
        n_influencers = int(round(self.influencer_prob * self.num_agents))
        all_nodes = list(G0.nodes())
        influencer_nodes = set()
        if n_influencers > 0 and hda_influence > 0:
            # Sort nodes by degree (descending)
            degree_sorted = sorted(all_nodes, key=lambda n: G0.degree[n], reverse=True)
            n_hda = int(round(hda_influence * n_influencers))
            n_hda = min(n_hda, n_influencers, len(all_nodes))
            hda_candidates = degree_sorted[:n_influencers]  # Top-n by degree
            # the top n nodes by degree are selected with probability hda_influence to be influencers
            hda_selected = set(random.sample(hda_candidates, n_hda)) if n_hda > 0 else set()
            influencer_nodes.update(hda_selected)
            # Remaining influencers are selected randomly from the remaining non-influencer nodes
            rest_candidates = [n for n in all_nodes if n not in hda_selected]
            n_rest = n_influencers - len(hda_selected)
            if n_rest > 0 and rest_candidates:
                rest_selected = set(random.sample(rest_candidates, min(n_rest, len(rest_candidates))))
                influencer_nodes.update(rest_selected)
        else:
            # Default: select n_influencers randomly
            influencer_nodes = set(random.sample(all_nodes, min(n_influencers, len(all_nodes)))) if n_influencers > 0 else set()
        flags = {node: (node in influencer_nodes) for node in all_nodes}
        self.influencer_flags = flags

        # For each possible edge, use a deterministic random generator based on seed, node pair, and edge type
        for i, j in G0.edges():
            i_inf = flags[i]
            j_inf = flags[j]
            # create a random number generator to use across all parameters set in model.py and agent.py
            # allows for replication across runs with the same seed
            edge_rng = random.Random(f"{seed}_{min(i,j)}_{max(i,j)}")
            if (i_inf and j_inf) or (not i_inf and not j_inf):
                # reciprocal edges
                try:
                    w = bounded_gamma(edge_rng, self.rec_max, 0.95)
                except ValueError as e: 
                    print(f"Error in bounded_gamma for reciprocal edge ({i}, {j}): {e}. Max is {self.rec_max}.")
                self.G.add_edge(i, j, weight=w)
                self.G.add_edge(j, i, weight=w)
            else:
                # one-way influencer→non-influencer
                if i_inf and not j_inf:
                    try:
                        w = bounded_gamma(edge_rng, self.one_max, 0.95)
                    except ValueError as e:
                        print(f"Error in bounded_gamma for one-way edge ({i}, {j}): {e}. Max is {self.one_max}.")
                    self.G.add_edge(i, j, weight=w)
                elif j_inf and not i_inf:
                    try:
                        w = bounded_gamma(edge_rng, self.one_max, 0.95)
                    except ValueError as e:
                        print(f"Error in bounded_gamma for one-way edge ({j}, {i}): {e}. Max is {self.one_max}.")
                    self.G.add_edge(j, i, weight=w)

        # Compute and store node positions ONCE for consistent visualization
        # Use only seed, num_agents, and sorted node list for layout seed
        node_list = tuple(sorted(G0.nodes()))
        layout_seed = hash((str(seed), num_agents, node_list)) % (2**32)
        # Use a fixed graph structure for layout: a simple graph with the same nodes, but no edges
        layout_graph = nx.Graph()
        layout_graph.add_nodes_from(node_list)
        self.pos = nx.spring_layout(layout_graph, seed=layout_seed)

        # Remove self-loops if any
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # Clean up: Remove any list-valued edge attributes (defensive fix for Mesa bug)
        for u, v, data in self.G.edges(data=True):
            for k in list(data.keys()):
                if isinstance(data[k], list):
                    # Remove or convert to float if possible
                    try:
                        data[k] = float(data[k][0])
                    except Exception:
                        del data[k]

        # Place agents
        self.grid = NetworkGrid(self.G)
        # For reproducibility: sort nodes so agent order is always the same for a given seed and num_agents
        sorted_nodes = sorted(self.G.nodes())
        for node in sorted_nodes:
            # Use a deterministic random generator for each agent's initial state
            agent_rng = random.Random(f"{seed}_{node}")
            agent = OpinionAgent(self, is_influencer=flags[node], rng=agent_rng)
            self.grid.place_agent(agent, node)

        # compute weighted average belief across neighboring nodes 
        def average_belief(m):
            beliefs = [a.belief for a in m.agents]
            if len(beliefs) == 0:
                return 0.0
            return np.mean(beliefs)

        def polarization_index(m):
            return np.var([a.belief for a in m.agents]) / 0.25

        model_reporters = {
            "AverageBelief": average_belief,
            "PolarizationIndex": polarization_index,
        }
        self.datacollector = DataCollector(model_reporters)
        self.running = True
        # Collect initial values for all model variables

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        for agent in self.agents:  # Use Mesa's built-in AgentSet
            agent.step()  # Explicitly activate each agent
        self.steps += 1  # Increment step counter