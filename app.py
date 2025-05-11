import solara
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from model import OpinionNetworkModel

@solara.component
def ConfigPanel():
    # UI controls for model parameters
    network_type, set_network_type = solara.use_state("erdos_renyi")
    solara.select(
        label="Network Topology",
        value=network_type,
        values=["erdos_renyi", "watts_strogatz", "barabasi_albert"],
        on_value=set_network_type,
    )

    influencer_prob, set_influencer_prob = solara.use_state(0.1)
    solara.slider(
        label="Influencer Probability",
        value=influencer_prob,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_influencer_prob,
    )

    self_weight_min, set_self_weight_min = solara.use_state(0.0)
    solara.slider(
        label="Self-Weight Min",
        value=self_weight_min,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_self_weight_min,
    )

    self_weight_max, set_self_weight_max = solara.use_state(0.5)
    solara.slider(
        label="Self-Weight Max",
        value=self_weight_max,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_self_weight_max,
    )

    reciprocal_min, set_reciprocal_min = solara.use_state(0.1)
    solara.slider(
        label="Reciprocal Edge Weight Min",
        value=reciprocal_min,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_reciprocal_min,
    )

    reciprocal_max, set_reciprocal_max = solara.use_state(0.5)
    solara.slider(
        label="Reciprocal Edge Weight Max",
        value=reciprocal_max,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_reciprocal_max,
    )

    one_way_min, set_one_way_min = solara.use_state(0.1)
    solara.slider(
        label="One-Way Edge Weight Min",
        value=one_way_min,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_one_way_min,
    )

    one_way_max, set_one_way_max = solara.use_state(0.5)
    solara.slider(
        label="One-Way Edge Weight Max",
        value=one_way_max,
        min=0.0,
        max=1.0,
        step=0.01,
        on_value=set_one_way_max,
    )

    num_agents, set_num_agents = solara.use_state(100)
    solara.number_input(
        label="Number of Agents",
        value=num_agents,
        min=1,
        max=1000,
        on_value=set_num_agents,
    )

    # Topology-specific parameters
    if network_type == "erdos_renyi":
        er_p, set_er_p = solara.use_state(0.1)
        solara.slider(
            label="Erdos-Renyi p",
            value=er_p,
            min=0.0,
            max=1.0,
            step=0.01,
            on_value=set_er_p,
        )
    elif network_type == "watts_strogatz":
        ws_k, set_ws_k = solara.use_state(4)
        solara.number_input(
            label="Watts-Strogatz k (neighbors)",
            value=ws_k,
            min=1,
            max=num_agents - 1,
            on_value=set_ws_k,
        )
        ws_p, set_ws_p = solara.use_state(0.1)
        solara.slider(
            label="Watts-Strogatz rewire p",
            value=ws_p,
            min=0.0,
            max=1.0,
            step=0.01,
            on_value=set_ws_p,
        )
    else:
        ba_m, set_ba_m = solara.use_state(2)
        solara.number_input(
            label="Barabási-Albert m (edges per new node)",
            value=ba_m,
            min=1,
            max=num_agents - 1,
            on_value=set_ba_m,
        )

    if solara.button("Run Model"):
        # Prepare topology args
        topo_args = {}
        if network_type == "erdos_renyi":
            topo_args = {"p": er_p}
        elif network_type == "watts_strogatz":
            topo_args = {"k": ws_k, "p_rewire": ws_p}
        else:
            topo_args = {"m": ba_m}

        # Instantiate the model
        model = OpinionNetworkModel(
            num_agents=num_agents,
            network_type=network_type,
            influencer_prob=influencer_prob,
            self_weight_range=(self_weight_min, self_weight_max),
            reciprocal_weight_range=(reciprocal_min, reciprocal_max),
            one_way_weight_range=(one_way_min, one_way_max),
            **topo_args,
        )
        # Run a few steps
        for _ in range(10):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        # Display final Polarization Index
        final_pol = df["PolarizationIndex"].iloc[-1]
        solara.markdown(f"**Polarization Index (final step):** {final_pol:.3f}")

        # Plot network with edge weights
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = nx.spring_layout(model.G)
        weights = [model.G[u][v]['weight'] for u, v in model.G.edges()]
        nx.draw_networkx_nodes(model.G, pos, ax=ax)
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
        nx.draw_networkx_labels(model.G, pos, ax=ax)
        for u, v in model.G.edges():
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid, f"{model.G[u][v]['weight']:.2f}", fontsize=8,
                    ha='center', va='center')
        solara.pyplot(fig)

    # Layout widgets
    widgets = [network_type, influencer_prob, self_weight_min, self_weight_max,
               reciprocal_min, reciprocal_max, one_way_min, one_way_max, num_agents]
    if network_type == "erdos_renyi":
        widgets.append(er_p)
    elif network_type == "watts_strogatz":
        widgets.extend([ws_k, ws_p])
    else:
        widgets.append(ba_m)
    widgets.append(solara.button("Run Model"))
    return solara.grid(widgets, columns=2)


# Correct way to define the Solara page
page = ConfigPanel

page