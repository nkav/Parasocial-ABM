import numpy as np
import pandas as pd
from model import OpinionNetworkModel
import secrets
import concurrent.futures
import os
import argparse
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

def get_fixed_params():
    return dict(
        num_agents=50,
        hda_influence=0.8,
        self_weight_max=2.0,
        reciprocal_max=2.0,
        one_way_max=2.0,
        bias_exp_lambda=100.0,
        ba_m=2,  # Barabási–Albert: number of edges to attach from a new node to existing nodes
    )

influencer_probs = np.arange(0, 1.01, 0.1)  # 0, 0.1, ..., 1.0
num_runs = 100
steps_per_run = 2000
fixed_params = get_fixed_params()
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def ensure_output_dir():
    os.makedirs('output', exist_ok=True)

def save_summary_plots(summary, param_name, y_names, out_prefix):
    for y in y_names:
        yerr = 1.96 * summary[f'std_{y}'] / summary['n_runs']**0.5
        plt.figure()
        plt.errorbar(summary[param_name], summary[f'avg_{y}'], yerr=yerr, fmt='o-', capsize=5)
        plt.xlabel(param_name)
        plt.ylabel(f'Average change in {y.replace("_", " ")}')
        plt.title(f'{y.replace("_", " ").capitalize()} vs {param_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{out_prefix}_{y}.png')
        plt.close()

def pval_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''

def annotate_significance(summary, results_df, groupby_col):
    sig_levels = []
    sig_levels_opinion = []
    for val in summary[groupby_col]:
        group = results_df[results_df[groupby_col] == val]
        t_pol, p_pol = ttest_rel(group['final_polarization'], group['start_polarization'])
        t_avg, p_avg = ttest_rel(group['final_avg_opinion'], group['start_avg_opinion'])
        sig_levels.append(pval_to_stars(p_pol))
        sig_levels_opinion.append(pval_to_stars(p_avg))
    summary['pol_sig'] = sig_levels
    summary['opinion_sig'] = sig_levels_opinion
    return summary

def batch_experiment(
    param_name, param_values, job_builder, simulation_fn, groupby_col, out_prefix
):
    jobs = []
    for val in param_values:
        seeds = set()
        while len(seeds) < num_runs:
            seeds.add(secrets.randbits(32))
        for seed in seeds:
            jobs.append(job_builder(val, seed))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(simulation_fn, jobs))
    results_df = pd.DataFrame(results)
    results_path = f'output/batch_results_{out_prefix}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Batch run complete. Results saved to {results_path}.")
    summary = results_df.groupby(groupby_col).agg(
        avg_polarization = ('polarization', 'mean'),
        std_polarization = ('polarization', 'std'),
        avg_avg_opinion = ('avg_opinion', 'mean'),
        std_avg_opinion = ('avg_opinion', 'std'),
        n_runs = ('seed', 'count'),
    ).reset_index()
    summary = annotate_significance(summary, results_df, groupby_col)
    summary_path = f'output/batch_summary_{out_prefix}.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}:")
    print(summary)
    save_summary_plots(
        summary,
        param_name=param_name,
        y_names=['polarization', 'avg_opinion'],
        out_prefix=out_prefix
    )

def run_experiment_vary_influencer_prob():
    batch_experiment(
        param_name='influencer_prob',
        param_values=influencer_probs,
        job_builder=lambda val, seed: (val, seed, fixed_params['reciprocal_max']),
        simulation_fn=run_simulation,
        groupby_col='influencer_prob',
        out_prefix='influencer_prob',
    )

def run_experiment_vary_reciprocal_max():
    reciprocal_max_values = np.arange(2, 10, 1)
    batch_experiment(
        param_name='reciprocal_max',
        param_values=reciprocal_max_values,
        job_builder=lambda val, seed: (0.1, seed, val),
        simulation_fn=run_simulation,
        groupby_col='reciprocal_max',
        out_prefix='reciprocal_max',
    )

def run_simulation_one_way(args):
    influencer_prob, seed, reciprocal_max, one_way_max = args
    params = dict(fixed_params)
    params['influencer_prob'] = influencer_prob
    params['reciprocal_max'] = reciprocal_max
    params['one_way_max'] = one_way_max
    model = OpinionNetworkModel(
        **params,
        seed=seed
    )
    for _ in range(steps_per_run):
        model.step()
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    pol = df['PolarizationIndex'].values
    avg = df['AverageBelief'].values
    result = {
        'start_polarization': pol[0],
        'final_polarization': pol[-1],
        'start_avg_opinion': avg[0],
        'final_avg_opinion': avg[-1],
        'avg_opinion': avg[-1] - avg[0],
        'polarization': pol[-1] - pol[0],
        'influencer_prob': influencer_prob,
        'reciprocal_max': reciprocal_max,
        'one_way_max': one_way_max,
        'seed': seed,
    }
    for thresh in thresholds:
        above = np.where(pol >= thresh)[0]
        result[f'step_pol_gt_{thresh}'] = int(above[0]) if len(above) > 0 else -1
    print(f"Done: influencer_prob={influencer_prob:.2f}, reciprocal_max={reciprocal_max}, one_way_max={one_way_max}, seed={seed}")
    return result

def run_experiment_vary_one_way_max():
    one_way_max_values = np.arange(2, 11, 1)
    batch_experiment(
        param_name='one_way_max',
        param_values=one_way_max_values,
        job_builder=lambda val, seed: (0.1, seed, 2.0, val),
        simulation_fn=run_simulation_one_way,
        groupby_col='one_way_max',
        out_prefix='one_way_max',
    )

def run_simulation(args):
    influencer_prob, seed, reciprocal_max = args
    params = dict(fixed_params)
    params['influencer_prob'] = influencer_prob
    params['reciprocal_max'] = reciprocal_max
    model = OpinionNetworkModel(
        **params,
        seed=seed
    )
    for _ in range(steps_per_run):
        model.step()
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    pol = df['PolarizationIndex'].values
    avg = df['AverageBelief'].values
    result = {
        'start_polarization': pol[0],
        'final_polarization': pol[-1],
        'start_avg_opinion': avg[0],
        'final_avg_opinion': avg[-1],
        'avg_opinion': avg[-1] - avg[0],
        'polarization': pol[-1] - pol[0],
        'avg_opinion': avg[-1] - avg[0],
        'influencer_prob': influencer_prob,
        'reciprocal_max': reciprocal_max,
        'seed': seed,
    }
    for thresh in thresholds:
        above = np.where(pol >= thresh)[0]
        result[f'step_pol_gt_{thresh}'] = int(above[0]) if len(above) > 0 else -1
    print(f"Done: influencer_prob={influencer_prob:.2f}, reciprocal_max={reciprocal_max}, seed={seed}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch experiments for opinion dynamics model.")
    parser.add_argument('--vary-influencer-prob', action='store_true', help='Run experiment varying influencer_prob')
    parser.add_argument('--vary-reciprocal-max', action='store_true', help='Run experiment varying reciprocal_max')
    parser.add_argument('--vary-one-way-max', action='store_true', help='Run experiment varying one_way_max')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()
    ensure_output_dir()
    if args.all or (not args.vary_influencer_prob and not args.vary_reciprocal_max and not args.vary_one_way_max):
        print("Running all experiments...")
        run_experiment_vary_influencer_prob()
        run_experiment_vary_reciprocal_max()
        run_experiment_vary_one_way_max()
    else:
        if args.vary_influencer_prob:
            run_experiment_vary_influencer_prob()
        if args.vary_reciprocal_max:
            run_experiment_vary_reciprocal_max()
        if args.vary_one_way_max:
            run_experiment_vary_one_way_max()
