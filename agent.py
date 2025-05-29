from mesa import Agent
from utils import bounded_gamma

class OpinionAgent(Agent):
    """
    Agent with continuous belief, influencer flag, self-weight,
    and bias attribute for biased assimilation (b_i ≥ 0) using Dandekar et al. equation
    """
    def __init__(self, model, is_influencer, rng=None):
        super().__init__(model)
        # Belief: normal distribution, mean 0.5, std 0.2
        belief = rng.gauss(0.5, 0.2)
        # ensure no values outside [0, 1]
        self.belief = min(1.0, max(0.0, belief))
        # Influencer flag
        self.is_influencer = is_influencer
        # Self-weight: right-tailed (gamma) with mode of 1, max set to 95 percentile of distribution
        self.self_weight = bounded_gamma(rng, model.self_weight_max, percentile=0.95)
        # Assimilation Bias: right-tailed (exponential) centered near 0, clipped to [0, 100]
        bias_lambda = getattr(model, 'bias_exp_lambda', 1.0)
        bias = rng.expovariate(bias_lambda)
        self.bias = min(100, max(0, bias))

    def step(self):
        # DeGroot's weighted average extension with bias
        w_ii = self.self_weight
        xi = self.belief
        # Compute s_i(t): weighted sum of neighbor beliefs
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        s_i = 0.0
        d_i = 0.0
        for n in neighbors:
            # Defensive: only use edge if it exists
            if self.model.G.has_edge(n.unique_id, self.unique_id):
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