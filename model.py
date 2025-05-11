import random
import networkx as nx
import numpy as np
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.agent import AgentSet  # New import for AgentSet
from agent import OpinionAgent

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
        network_type="erdos_renyi",
        influencer_prob=0.1,
        self_weight_range=(0.0, 0.5),
        reciprocal_weight_range=(0.1, 0.5),
        one_way_weight_range=(0.1, 0.5),
        p=None,
        k=None,
        p_rewire=None,
        m=None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.steps = 0  # Internal step counter
        self.agents = AgentSet()  # Replaces RandomActivation

        # Configurable parameters
        self.influencer_prob = influencer_prob
        self.self_weight_min, self.self_weight_max = self_weight_range
        self.rec_min, self.rec_max = reciprocal_weight_range
        self.one_min, self.one_max = one_way_weight_range

        # Build base undirected graph
        if network_type == "erdos_renyi":
            G0 = nx.erdos_renyi_graph(n=self.num_agents, p=p)
        elif network_type == "watts_strogatz":
            G0 = nx.watts_strogatz_graph(n=self.num_agents, k=k, p=p_rewire)
        elif network_type == "barabasi_albert":
            G0 = nx.barabasi_albert_graph(n=self.num_agents, m=m)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        # Prepare directed graph with weights
        self.G = nx.DiGraph()
        self.G.add_nodes_from(G0.nodes())

        # Assign influencer flags
        flags = {node: (random.random() < self.influencer_prob) for node in G0.nodes()}
        self.influencer_flags = flags

        # Convert edges with weight rules
        for i, j in G0.edges():
            i_inf = flags[i]
            j_inf = flags[j]
            if (i_inf and j_inf) or (not i_inf and not j_inf):
                # reciprocal edges
                w = random.uniform(self.rec_min, self.rec_max)
                self.G.add_edge(i, j, weight=w)
                self.G.add_edge(j, i, weight=w)
            else:
                # one-way influencer→non-influencer
                if i_inf and not j_inf:
                    w = random.uniform(self.one_min, self.one_max)
                    self.G.add_edge(i, j, weight=w)
                elif j_inf and not i_inf:
                    w = random.uniform(self.one_min, self.one_max)
                    self.G.add_edge(j, i, weight=w)

        # Place agents
        self.grid = NetworkGrid(self.G)
        for node in self.G.nodes():
            agent = OpinionAgent(node, self, is_influencer=flags[node])
            self.agents.add(agent)  # Add agent to AgentSet
            self.grid.place_agent(agent, node)

        # Data collector: weighted average belief
        def weighted_avg_belief(m):
            weights = []
            beliefs = []
            for a in m.agents:
                in_edges = m.G.in_edges(a.unique_id, data='weight')
                total_w = sum(w for _, _, w in in_edges)
                weights.append(total_w)
                beliefs.append(a.belief)
            weights = np.array(weights)
            if weights.sum() == 0:
                return np.mean(beliefs)
            return np.dot(weights, beliefs) / weights.sum()

        self.datacollector = DataCollector(
            {
                "WeightedAverageBelief": weighted_avg_belief,
                "PolarizationIndex": lambda m: np.var([a.belief for a in m.agents]) / 0.25,
            }
        )
        self.running = True

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        for agent in self.agents:
            agent.step()  # Explicitly activate each agent
        self.steps += 1  # Increment step counter