import solara
import matplotlib.pyplot as plt
import networkx as nx
from model import OpinionNetworkModel
from mesa.visualization import SolaraViz, make_plot_component
from solara import FigureMatplotlib

# Define model parameters for UI
model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": 100,
        "label": "Number of Agents",
        "min": 5,
        "max": 200,
        "step": 5,
    },
    "influencer_prob": {
        "type": "SliderFloat",
        "value": 0.1,
        "label": "Influencer Probability",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "self_weight_max": {
        "type": "SliderFloat",
        "value": 2.0,
        "label": "Self-Weight Max",
        "min": 2.0,
        "max": 10.0,
        "step": 1.0,
    },
    "reciprocal_max": {
        "type": "SliderFloat",
        "value": 2.0,
        "label": "Reciprocal Edge Weight Max",
        "min": 2.0,
        "max": 10.0,
        "step": 1.0
    },
    "one_way_max": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "One-Way Edge Weight Max",
        "min": 2.0,
        "max": 10.0,
        "step": 1.0,
    },
    "hda_influence": {
        "type": "SliderFloat",
        "value": 0.8,
        "label": "HDA Influence %",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "help": "Fraction of influencers chosen from highest-degree nodes (0=random, 1=all high-degree)",
    },
    "ba_m": {
        "type": "SliderInt",
        "value": 2,
        "label": "BA m (edges per new node)",
        "min": 1,
        "max": 10,
        "step": 1,
        "help": "Number of edges to attach from a new node to existing nodes in Barabási–Albert graph."
    },
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "bias_exp_lambda": {
        "type": "SliderFloat",
        "value": 100,
        "label": "Bias Exponential Lambda",
        "min": 10,
        "max": 1000,
        "step": 10,
        "help": "Lambda parameter for agent bias expovariate (mean=1/lambda)"
    },
}

# Visualization component for the network
def network_plot(model):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = model.pos
    weights = [model.G[u][v]['weight'] for u, v in model.G.edges()]
    # Get agent beliefs for color mapping
    agent_beliefs = {a.unique_id: a.belief for a in getattr(model, 'agents', [])}
    # Map opinions to colors: 0=red, 0.5=white, 1=blue
    def opinion_to_color(opinion):
        # Clamp to [0,1]
        opinion = max(0.0, min(1.0, opinion))
        if opinion <= 0.5:
            # Red to white: (1, 2*opinion, 2*opinion)
            r = 1.0
            g = 2 * opinion
            b = 2 * opinion
        else:
            # White to blue: (2*(1-opinion), 2*(1-opinion), 1)
            r = 2 * (1 - opinion)
            g = 2 * (1 - opinion)
            b = 1.0
        return (r, g, b)
    node_colors = [opinion_to_color(agent_beliefs.get(n, 0.5)) for n in model.G.nodes]
    influencer_flags = getattr(model, 'influencer_flags', {})
    influencer_nodes = [n for n, is_inf in influencer_flags.items() if is_inf]
    non_influencer_nodes = [n for n in model.G.nodes if not influencer_flags.get(n, False)]
    nx.draw_networkx_nodes(
        model.G, pos, nodelist=non_influencer_nodes, ax=ax, node_color=[node_colors[list(model.G.nodes).index(n)] for n in non_influencer_nodes],
        node_size=300, edgecolors='black', linewidths=1.0
    )
    # Draw influencers
    if influencer_nodes:
        nx.draw_networkx_nodes(
            model.G, pos, nodelist=influencer_nodes, ax=ax, node_color=[node_colors[list(model.G.nodes).index(n)] for n in influencer_nodes],
            node_size=300, edgecolors='gold', linewidths=4.0
        )
    nx.draw_networkx_edges(
        model.G,
        pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=10,
        width=[w * 2 for w in weights],
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
    )
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')         # Set the axes background to black
    ax.set_axis_off()

    # Add legend for influencer and opinion color
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Influencer',
               markerfacecolor='white', markeredgecolor='gold', markersize=15, linewidth=0, markeredgewidth=4),
        Line2D([0], [0], marker='o', color='w', label='Non-influencer',
               markerfacecolor='white', markeredgecolor='black', markersize=15, linewidth=0, markeredgewidth=1),
        mpatches.Patch(color=(1,0,0), label='Opinion = 0 (Red)'),
        mpatches.Patch(color=(1,1,1), label='Opinion = 0.5 (White)'),
        mpatches.Patch(color=(0,0,1), label='Opinion = 1 (Blue)'),
    ]
    # Place legend below the plot, centered
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              frameon=True, facecolor='black', edgecolor='white', fontsize=9, ncol=2, labelcolor='white')
    # Set all tick and axis label colors to white (for completeness, though axes are off)
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.close(fig)  # <-- Prevents too many open figures
    return fig

# Custom network plot component for Solara
def NetworkPlotComponent(model):
    return FigureMatplotlib(network_plot(model))

# Define a plot for the polarization index using Mesa's make_plot_component for reactivity
PolarizationPlot = make_plot_component({"PolarizationIndex": "tab:blue"})

# Add a plot for AverageBelief
AverageBeliefPlot = make_plot_component({"AverageBelief": "tab:orange"})

# --- Distribution extraction and plotting ---
def get_agent_distributions(model):
    biases = []
    beliefs = []
    self_weights = []
    for agent in getattr(model, 'agents', []):
        biases.append(agent.bias)
        beliefs.append(agent.belief)
        self_weights.append(agent.self_weight)
    return biases, beliefs, self_weights

def get_edge_distributions(model):
    recip_weights = []
    oneway_weights = []
    flags = getattr(model, 'influencer_flags', {})
    for u, v, data in model.G.edges(data=True):
        i_inf = flags.get(u, False)
        j_inf = flags.get(v, False)
        w = data['weight']
        # Reciprocal: both influencer or both non-influencer, and edge in both directions
        if ((i_inf and j_inf) or (not i_inf and not j_inf)) and model.G.has_edge(v, u) and u < v:
            recip_weights.append(w)
        elif (i_inf and not j_inf) or (j_inf and not i_inf):
            oneway_weights.append(w)
    return recip_weights, oneway_weights

# --- Plotting functions ---
def plot_bias_distribution(model):
    biases, _, _ = get_agent_distributions(model)
    fig, ax = plt.subplots()
    ax.hist(biases, bins=30, color='orange', alpha=0.7)
    ax.set_title('Agent Bias Distribution')
    ax.set_xlabel('Bias')
    ax.set_ylabel('Count')
    plt.close(fig)
    return fig

def plot_belief_distribution(model):
    _, beliefs, _ = get_agent_distributions(model)
    fig, ax = plt.subplots()
    ax.hist(beliefs, bins=30, color='blue', alpha=0.7)
    ax.set_title('Agent Belief Distribution')
    ax.set_xlabel('Belief')
    ax.set_ylabel('Count')
    plt.close(fig)
    return fig

def plot_selfweight_distribution(model):
    _, _, self_weights = get_agent_distributions(model)
    fig, ax = plt.subplots()
    ax.hist(self_weights, bins=30, color='purple', alpha=0.7)
    ax.set_title('Agent Self-Weight Distribution')
    ax.set_xlabel('Self-Weight')
    ax.set_ylabel('Count')
    plt.close(fig)
    return fig

def plot_degree_distribution(model):
    if hasattr(model, 'G'):
        degrees = [d for n, d in model.G.degree()]
        fig, ax = plt.subplots()
        ax.hist(degrees, bins=range(min(degrees), max(degrees)+2), color='teal', alpha=0.7, align='left')
        ax.set_title('Node Degree Distribution')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        plt.close(fig)
        return fig
    else:
        return None

def plot_reciprocal_weights(model):
    recip_weights, _ = get_edge_distributions(model)
    fig, ax = plt.subplots()
    ax.hist(recip_weights, bins=30, color='green', alpha=0.7)
    ax.set_title('Reciprocal Edge Weights')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')
    plt.close(fig)
    return fig

def plot_oneway_weights(model):
    _, oneway_weights = get_edge_distributions(model)
    fig, ax = plt.subplots()
    ax.hist(oneway_weights, bins=30, color='red', alpha=0.7)
    ax.set_title('One-way Edge Weights')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')
    plt.close(fig)
    return fig

# --- Solara components for each plot ---
def BiasDistributionComponent(model):
    return FigureMatplotlib(plot_bias_distribution(model))

def BeliefDistributionComponent(model):
    return FigureMatplotlib(plot_belief_distribution(model))

def SelfWeightDistributionComponent(model):
    return FigureMatplotlib(plot_selfweight_distribution(model))

def DegreeDistributionComponent(model):
    return FigureMatplotlib(plot_degree_distribution(model))

def ReciprocalWeightsComponent(model):
    return FigureMatplotlib(plot_reciprocal_weights(model))

def OnewayWeightsComponent(model):
    return FigureMatplotlib(plot_oneway_weights(model))

# Instantiate model (dummy, will be re-instantiated by SolaraViz)
model = OpinionNetworkModel(
    num_agents=model_params["num_agents"]["value"],
    influencer_prob=model_params["influencer_prob"]["value"],
    self_weight_max=model_params["self_weight_max"]["value"],
    reciprocal_max=model_params["reciprocal_max"]["value"],
    one_way_max=model_params["one_way_max"]["value"],
    ba_m=model_params["ba_m"]["value"],
    seed=model_params["seed"]["value"],
    hda_influence=model_params["hda_influence"]["value"],
    bias_exp_lambda=model_params["bias_exp_lambda"]["value"],
)

# SolaraViz page
page = SolaraViz(
    model,
    components=[
        NetworkPlotComponent,
        PolarizationPlot,
        AverageBeliefPlot,
        BiasDistributionComponent,
        BeliefDistributionComponent,
        SelfWeightDistributionComponent,
        ReciprocalWeightsComponent,
        OnewayWeightsComponent,
        DegreeDistributionComponent,
    ],
    model_params=model_params,
    name="Parasocial ABM",
)

page