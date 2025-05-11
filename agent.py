from mesa import Agent
import random
import math

class OpinionAgent(Agent):
    """
    Agent with continuous belief, influencer flag, self-weight,
    and bias attribute for biased assimilation (b_i ≥ 0) using Dandekar et al. equation
    """
    def __init__(self, unique_id, model, is_influencer, initial_belief=None):
        super().__init__(unique_id, model)
        # Initialize belief
        self.belief = initial_belief if initial_belief is not None else random.random()
        # Influencer flag
        self.is_influencer = is_influencer
        # Self-weight w_ii for update rule
        self.self_weight = random.uniform(
            model.self_weight_min, model.self_weight_max
        )
        # Assimilation bias b_i ≥ 0
        self.bias = random.random()

    def step(self):
        # DeGroot's weighted average extension with bias
        w_ii = self.self_weight
        xi = self.belief
        # Compute s_i(t): weighted sum of neighbor beliefs
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        s_i = 0.0
        d_i = 0.0
        for n in neighbors:
            w_ij = self.model.G[n.unique_id][self.unique_id]['weight']
            s_i += w_ij * n.belief
            d_i += w_ij
        # Bias exponent b_i
        b = self.bias
        # Numerator and denominator per Dandekar et al.
        xi_b = xi ** b
        one_minus_xi_b = (1 - xi) ** b
        num = w_ii * xi + xi_b * s_i
        denom = w_ii + xi_b * s_i + one_minus_xi_b * (d_i - s_i)
        if denom > 0:
            self.belief = num / denom