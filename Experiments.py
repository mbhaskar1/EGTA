from matplotlib.ticker import MultipleLocator

import EGTA
from GamutGame import GamutGame
import scipy.stats
from scipy.special import logit, expit
from scipy.stats import bootstrap
from statsmodels.distributions.empirical_distribution import ECDF
from NoisyGame import Noise, NoiseFactor, NoiseType, NoisyGame, ChancePlayer
from Algorithms import *
from Games import *
from matplotlib import pyplot as plt
import seaborn as sns
from EGTA import h, get_stats
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.ticker as mticker
from EmpiricalGame import EmpiricalGame
from BiddingGame import BiddingGame

game_gens = [
    (congestion_game, {'players': 3, 'facilities': 3, 'c': 2}, 'Congestion Game'),
    (war_of_attrition, {'actions': 18, 'c': 2}, 'War of Attrition'),
    (travelers_dilemma, {'players': 3, 'actions': 7, 'c': 2}, 'Travelers Dilemma'),
    (grab_the_dollar, {'actions': 18, 'c': 2}, 'Grab the Dollar'),
    (random_zero_sum, {'actions': 18, 'c': 2}, 'Random Zero Sum'),
    (covariant_game, {'players': 3, 'actions': 7, 'r': 0.85, 'c': 2}, 'Covariant Game'),
    (bertrand_oligopoly, {'players': 3, 'actions': 7, 'c': 2}, 'Bertrand Oligopoly'),
    (dispersion_game, {'players': 3, 'actions': 7, 'c': 2}, 'Dispersion Game')
]

congestion_game_gen = game_gens[0]
war_of_attrition_gen = game_gens[1]
travelers_dilemma_gen = game_gens[2]
grab_the_dollar_gen = game_gens[3]
random_zero_sum_gen = game_gens[4]
covariant_game_gen = game_gens[5]
bertrand_oligopoly_gen = game_gens[6]
dispersion_game_gen = game_gens[7]


def create_pandas_dataframe(df, x, new_data_hue_1, new_data_hue_2, x_category_name, data_category_name,
                            hue_category_name, hue_1_name, hue_2_name):
    df_empirical_ratio = pd.DataFrame({data_category_name: new_data_hue_1})
    df_empirical_ratio[hue_category_name] = hue_1_name
    df_non_empirical_ratio = pd.DataFrame({data_category_name: new_data_hue_2})
    df_non_empirical_ratio[hue_category_name] = hue_2_name

    df_ratios = pd.concat([df_empirical_ratio, df_non_empirical_ratio])
    df_ratios[x_category_name] = x

    if df is None:
        df = df_ratios
    else:
        df = pd.concat([df, df_ratios])
    return df


def gs_vs_psp_single_run(empirical_game: NoisyGame, target_epsilon, delta=0.05,
                         gs_batch_size=100, gs_initial_batch_size=100, psp_beta=1.5):
    c = empirical_game.get_c()

    gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                    target_epsilon=target_epsilon, batch_size=gs_batch_size,
                                                    initial_batch_size=gs_initial_batch_size, show_graphs_every=10)
    print('GS done')
    psp_sample_history, psp_results = progressive_sampling_with_pruning(
        empirical_game,
        c=c,
        delta=delta,
        target_epsilon=target_epsilon,
        beta=psp_beta,
        well_estimated_pruning=True,
        regret_pruning=True,
        show_graphs_every=-1)
    print('PSP done')

    gs_sup_epsilon = gs_results['supremum_epsilon']
    psp_sup_epsilon = psp_results['supremum_epsilon']
    psp_inf_epsilon = psp_results['infimum_epsilon']
    psp_num_active = psp_results['num_active_profiles']

    plt.plot(gs_sample_history[10:], gs_sup_epsilon[10:], color='red')
    plt.plot(psp_sample_history, psp_sup_epsilon, color='blue')
    plt.legend(['Global Sampling', 'Progressive Sampling w/ Pruning'])
    plt.xlabel('Num Queries / Game Size')
    plt.ylabel('SD Bound')
    plt.title('GS vs PSP')
    plt.axhline(y=target_epsilon, xmax=max(gs_sample_history[-1], psp_sample_history[-1]), color='green')
    plt.show()

    plt.plot(psp_num_active)
    plt.title('PSP Num Active Profiles')
    plt.xlabel('Iteration')
    plt.ylabel('Num Active')
    plt.show()

    plt.plot(psp_sample_history, psp_sup_epsilon, color='red')
    plt.plot(psp_sample_history, psp_inf_epsilon, color='blue')
    plt.legend(['Highest epsilon', 'Lowest epsilon'])
    plt.title('Non-Uniform Bounds in PSP')
    plt.xlabel('Num Queries / Game Size')
    plt.ylabel('Epsilon')
    plt.show()


def gs_vs_psp_sample_complexities(empirical_game: NoisyGame, target_epsilons, batch_sizes, delta=0.05,
                                  psp_beta=1.5):
    c = empirical_game.get_c()
    gs_sample_complexities = []
    psp_sample_complexities = []
    for batch_size, target_epsilon in zip(batch_sizes, target_epsilons):
        gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                        target_epsilon=target_epsilon, batch_size=batch_size,
                                                        show_graphs_every=-1)
        print('GS done')
        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=psp_beta,
            well_estimated_pruning=True,
            regret_pruning=True,
            show_graphs_every=-1)
        print('PSP done')

        gs_sample_complexities.append(gs_sample_history[-1])
        psp_sample_complexities.append(psp_sample_history[-1])

    print('plotting')
    plt.plot(target_epsilons, gs_sample_complexities, color='red')
    plt.plot(target_epsilons, psp_sample_complexities, color='blue')
    plt.legend(['GS', 'PSP'])
    plt.xlabel('Epsilon')
    plt.ylabel('Sample Complexity')
    plt.title('Sample Complexity vs Epsilon')
    plt.show()
    print('done')


def ps_vs_psp_sample_complexities(empirical_game: NoisyGame, target_epsilons, delta=0.05,
                                  beta=1.5):
    c = empirical_game.get_c()
    gs_sample_complexities = []
    ps_sample_complexities = []
    psp_sample_complexities = []
    for target_epsilon in target_epsilons:
        gs_sample_complexities.append(gs_h_sample_complexity(empirical_game, c, delta, target_epsilon))
        print(f'Target Epsilon: {target_epsilon}')
        ps_sample_history, ps_results = progressive_sampling(empirical_game, c, delta, target_epsilon, beta, verbose=0)
        print('PS done')
        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=beta,
            well_estimated_pruning=True,
            regret_pruning=True,
            show_graphs_every=-1, verbose=0)
        print('PSP done')

        ps_sample_complexities.append(ps_sample_history[-1])
        psp_sample_complexities.append(psp_sample_history[-1])

    print('plotting')
    plt.plot(target_epsilons, gs_sample_complexities, color='brown')
    plt.plot(target_epsilons, ps_sample_complexities, color='red')
    plt.plot(target_epsilons, psp_sample_complexities, color='blue')
    plt.legend(['GS', 'PS', 'PSP'])
    plt.xlabel('Epsilon')
    plt.ylabel('Sample Complexity')
    plt.title('Sample Complexity vs Epsilon')
    plt.show()
    print('done')


# Experiment results look most interesting for small congestion games
def frequency_of_eps_pure_equilibria(empirical_game: NoisyGame, delta=0.05, sample_sizes=None, num_algo_runs=200):
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500]
    assert len(sample_sizes) == 4

    c = empirical_game.get_c()
    utils_shape = empirical_game.get_utils_shape()

    fig, axs = plt.subplots(2, 2, dpi=200)

    for idx, sample_size in enumerate(sample_sizes):
        eps_nash_frequency = np.zeros(utils_shape[:-1])
        for run in range(num_algo_runs):
            gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                            target_epsilon=0,
                                                            max_iterations=1,
                                                            batch_size=sample_size,
                                                            show_graphs_every=-1)
            empirical_utils = gs_results['final_empirical_utils']
            epsilon = gs_results['supremum_epsilon'][-1]
            eps_nash_frequency += eps_nash_equilibria(empirical_utils, 2 * epsilon)

        print(idx)
        eps_nash_frequency = np.maximum(1, eps_nash_frequency)
        eps_nash_frequency = sorted(eps_nash_frequency.reshape(-1).tolist())
        print(eps_nash_frequency)

        row = idx // 2
        col = idx % 2

        axs[row, col].bar(list(range(1, len(eps_nash_frequency) + 1)), eps_nash_frequency, width=0.65)
        axs[row, col].set_xticks(list(range(1, len(eps_nash_frequency) + 1)))
        axs[row, col].set_xticklabels([])
        axs[row, col].set_xlabel(f'{sample_size} Samples')

    fig.suptitle('Frequency of 2ϵ-Nash Equilibria')
    plt.show()


def psp_frequency_of_eps_pure_equilibria(game_class, game_params: dict, empirical_game_params: dict, iterations=None,
                                         num_games=200,
                                         delta=0.05,
                                         target_epsilon: float = 5, beta=1.5, queries_scale=4):
    if iterations is None:
        iterations = [1, 2, 3, 4]
    assert len(iterations) == 4

    rows = 1
    cols = 4
    fig, axs = plt.subplots(rows, cols, dpi=200, figsize=(16, 4.65))

    # Generate first empirical game, and get relevant information
    empirical_game = construct_empirical_game(game_class(**game_params), **empirical_game_params)
    c = empirical_game.get_c()
    num_profiles = np.prod(empirical_game.get_utils_shape()[:-1])

    eps_nash_frequencies = []
    true_nash_frequency = np.zeros(num_profiles)
    queries_total = []
    for i in range(4):
        eps_nash_frequencies.append(np.zeros(num_profiles))
    for game_num in range(num_games):
        while True:
            true_regrets = np.amax(regrets_matrix(empirical_game.expected_utilities()), axis=-1).reshape(-1)
            regret_sorted_indices = np.flip(np.argsort(true_regrets))
            true_nash = (true_regrets == 0).astype(int)
            if np.count_nonzero(true_nash) != 1:
                print('BAD GAME - SKIPPING')
                empirical_game = construct_empirical_game(game_class(**game_params), **empirical_game_params)
            else:
                break

        psp_sample_history, psp_results = progressive_sampling_with_pruning(empirical_game, c=c, delta=delta,
                                                                            target_epsilon=target_epsilon,
                                                                            max_iterations=max(iterations), beta=beta,
                                                                            return_intermediate_utils=True,
                                                                            return_epsilon_matrices=True,
                                                                            old_regret_pruning=True,
                                                                            verbose=2)
        empirical_utils = psp_results['empirical_utils']
        epsilon_matrices = psp_results['epsilon_matrices']
        num_active_profiles = psp_results['num_active_profiles']

        queries = []

        for i in range(len(psp_sample_history)):
            if len(queries) > 0:
                queries.append(
                    queries[i - 1] + (psp_sample_history[i] - psp_sample_history[i - 1]) * num_active_profiles[i])
            else:
                queries.append(psp_sample_history[0])

        queries_total.append(queries)

        for i in range(4):
            iteration = iterations[i]
            eps_nash_frequencies[i] += (lower_bound_epsilon_nash_equilibria(empirical_utils[iteration - 1],
                                                                            epsilon_matrices[iteration - 1]).reshape(-1)
                                        )[regret_sorted_indices]

        print(game_num)

        # Generate new game
        empirical_game = construct_empirical_game(game_class(**game_params), **empirical_game_params)

    true_nash_frequency = true_nash_frequency.tolist()
    print(true_nash_frequency)

    queries = np.round(np.average(queries_total, axis=0)).tolist()

    for idx in range(4):
        eps_nash_frequency = eps_nash_frequencies[idx]
        # eps_nash_frequency = np.maximum(1, eps_nash_frequency)
        eps_nash_frequency = eps_nash_frequency.tolist()

        row = idx // cols
        col = idx % cols

        if rows > 1 and cols > 1:
            ax = axs[row, col]
        elif cols > 1:
            ax = axs[col]
        elif rows > 1:
            ax = axs[rows]
        else:
            ax = axs

        ax.fill_between(np.arange(0, len(eps_nash_frequency) + 1, 0.01), np.amax(eps_nash_frequency[:-1]), num_games,
                        color='lightgreen')
        ax.axhline(np.amax(eps_nash_frequency[:-1]), linestyle='--', color='seagreen')
        ax.bar(list(range(1, len(eps_nash_frequency) + 1)), eps_nash_frequency, width=0.65,
               color=['royalblue' if i != len(eps_nash_frequency) - 1 else 'orangered' for i in
                      range(len(eps_nash_frequency))]
               )
        ax.set_xticks(list(range(1, len(eps_nash_frequency) + 1)))
        ax.set_xticklabels([])
        iteration = iterations[idx]
        ax.set_xlabel(f'{iteration} Iteration{"" if iteration == 1 else "s"}; '
                      f'{queries[iteration - 1] / (10 ** queries_scale):.2f}'
                      f'$\\times 10^{queries_scale}$ Queries on Avg.')
        if row == 0 and col == 0:
            ax.set_ylabel('Num Games')
        ax.set_xlim(0, len(eps_nash_frequency) + 1)
        ax.set_ylim(0, num_games)

    fig.suptitle('Frequency of Candidate Nash Equilibria in PSP')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

    with open('frequencies.txt', 'w') as f:
        for i in range(4):
            f.write(np.array2string(eps_nash_frequencies[i]))
            f.write('\n')


def gs_frequency_of_eps_pure_equilibria(game_gen_func, sample_sizes=None,
                                        num_games=200,
                                        delta=0.05,
                                        batch_size=10000,
                                        require_one_nash=False,
                                        second_inclusion=False,
                                        smart_candidate_eqa=False):
    if sample_sizes is None:
        sample_sizes = [30000, 100000, 200000, 350000]
    assert len(sample_sizes) == 4

    rows = 1
    cols = 4
    fig, axs = plt.subplots(rows, cols, dpi=200, figsize=(16, 4.65))

    # Generate first empirical game, and get relevant information
    empirical_game = game_gen_func()
    c = empirical_game.get_c()
    num_profiles = np.prod(empirical_game.get_utils_shape()[:-1])

    eps_nash_frequencies = []
    two_eps_nash_frequencies = []
    true_nash_frequency = np.zeros(num_profiles)
    for i in range(4):
        eps_nash_frequencies.append(np.zeros(num_profiles))
        two_eps_nash_frequencies.append(np.zeros(num_profiles))
    for game_num in range(num_games):
        while True:
            true_utils = empirical_game.expected_utilities()
            true_regrets = np.amax(regrets_matrix(true_utils), axis=-1).reshape(-1)
            regret_sorted_indices = np.flip(np.argsort(true_regrets))
            true_nash = np.sort((true_regrets == 0).astype(int))
            if require_one_nash and np.count_nonzero(true_nash) != 1:
                print('BAD GAME - SKIPPING')
                empirical_game = game_gen_func()
            else:
                true_nash_frequency += true_nash
                break

        for idx in range(4):
            sample_size = sample_sizes[idx]

            gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                            target_epsilon=0,
                                                            max_iterations=int(np.ceil(sample_size / batch_size)),
                                                            batch_size=batch_size,
                                                            show_graphs_every=-1,
                                                            verbose=2)

            empirical_utils = gs_results['final_empirical_utils']
            epsilon_matrix = gs_results['final_epsilon_matrix']
            sup_eps = np.amax(epsilon_matrix)

            if smart_candidate_eqa:
                eps_nash_frequencies[idx] += (lower_bound_epsilon_nash_equilibria(empirical_utils,
                                                                                  epsilon_matrix).reshape(-1)
                                              )[regret_sorted_indices]
                two_eps_nash_frequencies[idx] += (lower_bound_epsilon_nash_equilibria(empirical_utils,
                                                                                      2 * epsilon_matrix).reshape(-1)
                                                  )[regret_sorted_indices]
            else:
                eps_nash_frequencies[idx] += (eps_nash_equilibria(empirical_utils, 2 * sup_eps).reshape(-1)
                                              )[regret_sorted_indices]
                two_eps_nash_frequencies[idx] += (eps_nash_equilibria(empirical_utils, 4 * sup_eps).reshape(-1)
                                                  )[regret_sorted_indices]
            print(f'1: {np.count_nonzero(eps_nash_frequencies[idx] >= true_nash_frequency)}')
            print(f'2: {np.count_nonzero(two_eps_nash_frequencies[idx] >= eps_nash_frequencies[idx])}')

        print(game_num)

        # Generate new game
        empirical_game = game_gen_func()

    for idx in range(4):
        eps_nash_frequency = eps_nash_frequencies[idx]
        two_eps_nash_frequency = two_eps_nash_frequencies[idx]

        red_blue_sum_frequency = eps_nash_frequency.tolist()

        two_eps_nash_frequency -= eps_nash_frequency
        eps_nash_frequency -= true_nash_frequency
        # eps_nash_frequency = np.maximum(1, eps_nash_frequency)
        eps_nash_frequency = eps_nash_frequency.tolist()
        two_eps_nash_frequency = two_eps_nash_frequency.tolist()

        row = idx // cols
        col = idx % cols

        if rows > 1 and cols > 1:
            ax = axs[row, col]
        elif cols > 1:
            ax = axs[col]
        elif rows > 1:
            ax = axs[rows]
        else:
            ax = axs

        ax.fill_between(np.arange(0, len(eps_nash_frequency) + 1, 0.01), np.amax(eps_nash_frequency[:-1]), num_games,
                        color='lightgreen')
        ax.axhline(np.amax(eps_nash_frequency[:-1]), linestyle='--', color='seagreen')
        if second_inclusion:
            ax.bar(list(range(1, len(two_eps_nash_frequency) + 1)), two_eps_nash_frequency, width=0.65, color='yellow',
                   label='Eliminated True $4\\hat{\\epsilon}$-Nash Equilibria', bottom=red_blue_sum_frequency)
        ax.bar(list(range(1, len(eps_nash_frequency) + 1)), eps_nash_frequency, width=0.65, color='royalblue',
               label='Spurious Equilibria',
               bottom=true_nash_frequency)
        ax.bar(list(range(1, len(true_nash_frequency) + 1)), true_nash_frequency, width=0.65, color='orangered',
               label='True Nash Equilibria')
        ax.set_xticks(list(range(1, len(eps_nash_frequency) + 1)))
        ax.set_xticklabels([])
        ax.set_title(f'{sample_sizes[idx]} Samples')
        ax.set_xlabel('Strategy Profiles')
        if row == 0 and col == 0:
            ax.set_ylabel('Num Games')
        ax.set_xlim(0, len(eps_nash_frequency) + 1)
        ax.set_ylim(0, num_games)

        ax.legend().set_visible(False)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3 if second_inclusion else 2)

    fig.suptitle('Frequency of Candidate Nash Equilibria')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

    with open('frequencies.txt', 'w') as f:
        f.write(np.array2string(true_nash_frequency))
        for i in range(4):
            f.write(np.array2string(eps_nash_frequencies[i]))
            f.write('\n')


def frequency_of_eps_pure_equilibria_by_batch(empirical_game: NoisyGame, delta=0.05, batch_size=50,
                                              num_iterations=100, num_algo_runs=200):
    c = empirical_game.get_c()
    utils_shape = empirical_game.get_utils_shape()

    streaming_object = [None] * num_algo_runs
    for n in range(1, num_iterations + 1):
        eps_nash_frequency = np.zeros(utils_shape[:-1])
        epsilon = None
        for run in range(num_algo_runs):
            gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                            target_epsilon=0,
                                                            max_iterations=1,
                                                            batch_size=batch_size,
                                                            streaming_object=streaming_object[run],
                                                            show_graphs_every=-1)
            empirical_utils = gs_results['final_empirical_utils']
            streaming_object[run] = gs_results['streaming_object']
            epsilon = gs_results['supremum_epsilon'][-1]
            eps_nash_frequency += eps_nash_equilibria(empirical_utils, 2 * epsilon)

        print(epsilon)
        eps_nash_frequency = np.maximum(1, eps_nash_frequency)
        eps_nash_frequency = sorted(eps_nash_frequency.reshape(-1).tolist())
        print(eps_nash_frequency)

        plt.bar(list(range(1, len(eps_nash_frequency) + 1)), eps_nash_frequency, width=0.45)
        plt.xticks(list(range(1, len(eps_nash_frequency) + 1)), [])
        plt.xlabel(f'{batch_size * n} Samples')
        plt.title('Frequency of 2ϵ-Nash Equilibria')
        plt.show()


def precision_recall_by_batch(game_gen_func,
                              delta=0.05,
                              batch_size=1000,
                              num_batches=10,
                              num_games=1):
    # Generate first empirical game, and get relevant information

    precision = []
    recall = []

    for game_num in range(num_games):
        print(game_num)
        empirical_game = game_gen_func()
        c = empirical_game.get_c()

        true_utils = empirical_game.expected_utilities()
        true_regrets = np.amax(regrets_matrix(true_utils), axis=-1).reshape(-1)
        true_nash = (true_regrets == 0)
        num_nash = np.count_nonzero(true_nash)

        streaming_object = None
        for idx in range(num_batches):
            gs_sample_history, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                                            target_epsilon=0,
                                                            max_iterations=1,
                                                            batch_size=batch_size,
                                                            streaming_object=streaming_object,
                                                            show_graphs_every=-1,
                                                            verbose=0)

            streaming_object = gs_results['streaming_object']
            empirical_utils = gs_results['final_empirical_utils']
            epsilon_matrix = gs_results['final_epsilon_matrix']

            eps_nash = (lower_bound_epsilon_nash_equilibria(empirical_utils, epsilon_matrix).reshape(-1))

            num_intersect = np.count_nonzero(np.multiply(true_nash, eps_nash))
            num_eps_nash = np.count_nonzero(eps_nash)

            precision.append(num_intersect / num_eps_nash)
            recall.append(num_intersect / num_nash)

    x = [batch_size * i for _ in range(num_games) for i in range(1, num_batches + 1)]
    sns.lineplot(x, precision, label='precision')
    sns.lineplot(x, recall, label='recall')
    plt.xlabel('Sample Size')
    plt.legend()
    plt.show()


def variance_sample_complexity():
    c = 10
    delta = 0.05
    eps = 0.5
    variance = np.linspace(0, c ** 2 / 4, 100)
    variance = variance[1:1]
    game_size = 5000
    log_term = np.log(2) + np.log(game_size) - np.log(delta)
    sample_complexity = log_term * (c ** 2 / (variance * h(c * eps / variance)))
    plt.plot(variance, sample_complexity)
    plt.show()


def regrets_distributions_by_batch(empirical_game: NoisyGame, sample_sizes, sample_size_labels=None, delta=0.05):
    c = empirical_game.get_c()
    if sample_size_labels is None:
        sample_size_labels = sample_sizes
    batch_sizes = [sample_sizes[0]]
    for n in range(1, len(sample_sizes)):
        batch_sizes.append(sample_sizes[n] - sample_sizes[n - 1])
    print(batch_sizes)

    df = None

    streaming_object = None
    num_iterations = len(sample_sizes)
    for n in range(1, num_iterations + 1):
        _, gs_results = global_sampling(empirical_game, c=c, delta=delta,
                                        target_epsilon=0,
                                        max_iterations=1,
                                        batch_size=batch_sizes[n - 1],
                                        streaming_object=streaming_object,
                                        show_graphs_every=-1)
        empirical_utils = gs_results['final_empirical_utils']
        epsilon_matrix = gs_results['final_epsilon_matrix']
        regrets_upper = regrets_upper_bound_matrix(empirical_utils, epsilon_matrix).reshape(-1)
        regrets = regrets_matrix(empirical_utils).reshape(-1)
        regrets_lower = regrets_lower_bound_matrix(empirical_utils, epsilon_matrix).reshape(-1)

        streaming_object = gs_results['streaming_object']

        df_regrets_upper = pd.DataFrame({'Regrets': regrets_upper})
        df_regrets_upper['Bound Type'] = 'upper'
        df_regrets_lower = pd.DataFrame({'Regrets': regrets_lower})
        df_regrets_lower['Bound Type'] = 'lower'

        df_regrets = pd.concat([df_regrets_upper, df_regrets_lower])
        df_regrets['Sample Size'] = sample_size_labels[n - 1]

        if df is None:
            df = df_regrets
        else:
            df = pd.concat([df, df_regrets])

        sns.violinplot(x='Sample Size', y='Regrets', hue='Bound Type', data=df, scale='count', split=True, bw=0.2,
                       inner='quartile')
        plt.show()


def regrets_distributions_psp(empirical_game: NoisyGame, delta=0.05, target_epsilon=0.05,
                              beta=1.1):
    c = empirical_game.get_c()

    df = None

    sample_history, results = progressive_sampling_with_pruning(empirical_game, c, delta, target_epsilon, beta,
                                                                regret_pruning=False, return_intermediate_utils=True,
                                                                return_epsilon_matrices=True)
    num_active_profiles = results['num_active_profiles']
    queries = None

    empirical_utils_matrices = results['empirical_utils']
    epsilon_matrices = results['epsilon_matrices']
    assert len(empirical_utils_matrices) == len(epsilon_matrices)

    for i in range(len(empirical_utils_matrices)):
        empirical_utils = empirical_utils_matrices[i]
        epsilon_matrix = epsilon_matrices[i]

        if queries is not None:
            queries += (sample_history[i] - sample_history[i - 1]) * num_active_profiles[i]
        else:
            queries = sample_history[0]

        regrets_upper = regrets_upper_bound_matrix(empirical_utils, epsilon_matrix).reshape(-1)
        regrets_lower = regrets_lower_bound_matrix(empirical_utils, epsilon_matrix).reshape(-1)

        df_regrets_upper = pd.DataFrame({'Regrets': regrets_upper})
        df_regrets_upper['Bound Type'] = 'upper'
        df_regrets_lower = pd.DataFrame({'Regrets': regrets_lower})
        df_regrets_lower['Bound Type'] = 'lower'

        df_regrets = pd.concat([df_regrets_upper, df_regrets_lower])
        df_regrets['Query Complexity'] = queries

        if df is None:
            df = df_regrets
        else:
            df = pd.concat([df, df_regrets])

        sns.violinplot(x='Query Complexity', y='Regrets', hue='Bound Type', data=df, scale='count', split=True, bw=0.2,
                       inner='quartile')
        plt.show()


def true_regret_distribution(game: Union[GamutGame, EmpiricalGame], title=None, show_graph=True):
    if isinstance(game, GamutGame):
        utils = game.get_utils()
    elif isinstance(game, EmpiricalGame):
        utils = game.expected_utils()
    else:
        print('ERROR')
        return
    print(utils)
    print(utils.shape)
    regrets = regrets_matrix(utils).reshape(-1)
    print(regrets)
    if show_graph:
        sns.ecdfplot(regrets)
        plt.title(title)
        plt.show()
    return regrets


def average_true_regret_distribution(game_func, params, num_games, num_individual_cdfs=None, remove_zero_regret=False,
                                     title=None, ax=None):
    regrets = []
    c = params['c']
    x = np.linspace(0, c, 200)
    ecdfs = []
    for n in range(num_games):
        regrets.append(true_regret_distribution(game_func(**params), show_graph=False).tolist())
        if remove_zero_regret:
            regrets[-1] = list(filter(lambda x: x != 0, regrets[-1]))
        ecdfs.append(ECDF(regrets[-1])(x))
        if num_individual_cdfs is None or n < num_individual_cdfs:
            sns.ecdfplot(regrets[-1], color='lightblue', ax=ax, label='Sampled Games' if n == 0 else None)
    regrets = list(chain(*regrets))
    all_x = np.tile(x, len(ecdfs)).tolist()
    all_ecdfs = list(chain(*ecdfs))
    np_ecdfs = np.array(ecdfs)

    max_ecdf = np.amax(np_ecdfs, axis=0)
    min_ecdf = np.amin(np_ecdfs, axis=0)

    # sns.ecdfplot(regrets, ax=ax)
    sns.lineplot(x, max_ecdf, color='salmon', drawstyle='steps-post', ax=ax, label='Min/Max Proportion')
    sns.lineplot(x, min_ecdf, color='salmon', drawstyle='steps-post', ax=ax)
    sns.lineplot(all_x, all_ecdfs, drawstyle='steps-post', ax=ax, label='Avg Proportion')
    if ax is None:
        plt.title(title)
        plt.xlabel('Epsilon')
        plt.show()
    else:
        ax.set_title(title)
        ax.set_xlabel('Epsilon')


def average_regret_distribution_many_games(num_game_samples=10, num_individual_cdfs=5, remove_zero_regret=False):
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(12, 6))
    for n in range(8):
        print(f'Game {n + 1}')
        row = n // 4
        col = n % 4
        game_func, params, title = game_gens[n]
        average_true_regret_distribution(game_func, params, num_game_samples, num_individual_cdfs=num_individual_cdfs,
                                         title=title, ax=axes[row, col],
                                         remove_zero_regret=remove_zero_regret)
        axes[row, col].set_xlim(0, 2)
    plt.tight_layout()
    plt.show()


def average_regret_distribution_many_games_amy(num_game_samples=100, num_individual_cdfs=5, remove_zero_regret=False):
    fig, axes = plt.subplots(2, 3, sharex=False, figsize=(9, 6))
    amy_game_gens = [
        random_zero_sum_gen, congestion_game_gen, grab_the_dollar_gen,
        covariant_game_gen, bertrand_oligopoly_gen, dispersion_game_gen
    ]
    for n in range(6):
        print(f'Game {n + 1}')
        row = n // 3
        col = n % 3
        game_func, params, title = amy_game_gens[n]
        average_true_regret_distribution(game_func, params, num_game_samples, num_individual_cdfs=num_individual_cdfs,
                                         title=title, ax=axes[row, col],
                                         remove_zero_regret=remove_zero_regret)
        axes[row, col].set_xlim(0, 2)
        axes[row, col].legend().set_visible(False)
        if row == 0 and col == 0:
            handles, labels = axes[row, col].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    fig.savefig('average_eps_nash_greenwald.pdf', bbox_inches='tight')


def proportion_pruned_vs_samples(empirical_game: NoisyGame, delta=0.05, beta=1.1, target_epsilon=5.0,
                                 num_iterations=1, title=None, ax=None):
    c = empirical_game.get_c()
    T = np.ceil(np.log(3 * c / (4 * target_epsilon)) / np.log(beta)).astype('int')
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players

    # log_term = np.log(2) + np.log(game_size) + np.log(T) - np.log(delta)
    # multiplier = 2 * log_term / (target_epsilon ** 2)
    # start_x = multiplier * c * target_epsilon / 3

    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    log_term_bennett = log_term - np.log(3) + np.log(2)
    start_x = 1 + (8.0 / 3 + np.sqrt(4 + 4 / log_term)) * c * log_term / target_epsilon
    multiplier = 2 * log_term / (target_epsilon ** 2)

    variance = empirical_game.variance_utils()
    print(c)
    bennett_bound = log_term_bennett * (c ** 2 / (variance * h(c * target_epsilon / variance)))
    bennett_bound = np.amax(bennett_bound, axis=-1)  # Only care about profiles
    bennett_bound = bennett_bound.reshape(-1)
    empirical_bennett_bound = 1 + beta * ((8.0 / 3 + np.sqrt(
        4 + 2 / log_term)) * c * log_term / target_epsilon + 2 * variance * log_term / target_epsilon ** 2)
    empirical_bennett_bound = np.amax(empirical_bennett_bound, axis=-1)  # Only care about profiles
    empirical_bennett_bound = empirical_bennett_bound.reshape(-1)
    var_ticks = 0.05
    max_sample_complexity = np.amax(empirical_bennett_bound)
    max_var = np.amax(variance)
    print(max_var)
    if max_var % var_ticks != 0:
        max_var += (var_ticks - max_var % var_ticks)

    sample_history = []
    proportion_pruned = []

    for n in range(num_iterations):
        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=beta,
            well_estimated_pruning=True,
            regret_pruning=False,
            show_graphs_every=-1)

        num_active_profiles = psp_results['num_active_profiles'][1:]
        proportion_pruned.extend(
            [1 - active_profiles / num_profiles for active_profiles in num_active_profiles])
        sample_history.extend(psp_sample_history)
        print(f'N: {n}')

    if ax is not None:
        ax.set_ylabel('Proportion Pruned')
        bennett_ecdf = ECDF(bennett_bound)
        e_bennett_ecdf = ECDF(empirical_bennett_bound)
        ax.fill_between(bennett_ecdf.x, bennett_ecdf.y, color='lightblue', step='post')
        ax.fill_between([0, *sample_history], [0, *proportion_pruned], color='palegreen', step='post')
        ax.fill_between(e_bennett_ecdf.x, e_bennett_ecdf.y, color='mistyrose', step='post')
        sns.ecdfplot(bennett_bound, label='Bennett (Known Variance)', ax=ax, color='royalblue')
        sns.lineplot([0, *sample_history], [0, *proportion_pruned], color='darkgreen', alpha=0.8,
                     drawstyle='steps-post', ax=ax)
        sns.scatterplot(sample_history, proportion_pruned, color='darkgreen', label='PSP', ax=ax)
        sns.ecdfplot(empirical_bennett_bound, label='PSP Data Complexity Upper Bound', ax=ax, color='tomato')
        ax.set_xlim(0, max_sample_complexity)
        ax.set_xlabel('Data Complexity')
        if title is not None:
            ax.set_title(title)
    else:
        plt.figure()
        plt.ylabel('Proportion Pruned')

        plt.subplot(111)
        sns.ecdfplot(bennett_bound)
        sns.scatterplot(sample_history, proportion_pruned)
        sns.ecdfplot(empirical_bennett_bound)
        plt.legend(
            ['Bennett (Known Variance)', 'PSP', 'PSP Data Complexity Upper Bound'])
        plt.xlim(0, max_sample_complexity)
        plt.xlabel('Data Complexity')
        if title is not None:
            plt.title(title)
        plt.show()

    # print(len(psp_sample_history))
    # print(len(proportion_pruned))
    #
    # ax.twiny()
    # sns.lineplot(psp_sample_history, proportion_pruned)
    # plt.xlim(start_x, start_x + max_var * multiplier)
    # plt.xlabel('Sample Complexity')
    # ax2.spines['top'].set_position(('axes', -0.15))
    # ax2.spines['top'].set_visible(False)
    # plt.tick_params(which='both', top=False)


def proportion_pruned_vs_samples_subplots():
    n_rows = 1
    n_cols = 4
    figsize = (16, 4.65)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    alphas = [0.5, 1.5, 3, 5]
    betas = [3, 3, 1.5, 0.5]
    titles = [
        'Beta(0.5, 3) — Mostly low variance',
        'Beta(1.5, 3) — Mostly medium variance',
        'Beta(3, 1.5) — Mostly high variance',
        'Beta(5, 0.5) — Almost all high variance'
    ]
    max_x_lim = 0
    for i in range(4):
        row = i // n_cols
        col = i % n_cols

        if row > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        proportion_pruned_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
                                                              complete_noise=10,
                                                              noise_distribution=scipy.stats.bernoulli,
                                                              noise_args={'p': 0.5},
                                                              noise_multiplier_distribution=scipy.stats.beta,
                                                              noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                                     'a': alphas[i],
                                                                                     'b': betas[i]}),
                                     target_epsilon=0.2, beta=1.1, title=titles[i], ax=ax)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(4):
        row = i // n_cols
        col = i % n_cols
        if row > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_xlim(0, max_x_lim)
        ax.legend().set_visible(False)
        if i == 3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    fig.savefig('proportion_pruned.pdf', bbox_inches='tight')


def proportion_remaining_vs_samples(empirical_game: EmpiricalGame, delta=0.05, beta=1.5, target_epsilon=5.0,
                                    num_iterations=1, title=None, ax=None):
    c = empirical_game.get_c()
    T = np.ceil(np.log(3 * c / (4 * target_epsilon)) / np.log(beta)).astype('int')
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players

    # log_term = np.log(2) + np.log(game_size) + np.log(T) - np.log(delta)
    # multiplier = 2 * log_term / (target_epsilon ** 2)
    # start_x = multiplier * c * target_epsilon / 3

    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    log_term_bennett = log_term - np.log(3) + np.log(2)
    start_x = 1 + (8.0 / 3 + np.sqrt(4 + 4 / log_term)) * c * log_term / target_epsilon
    multiplier = 2 * log_term / (target_epsilon ** 2)

    variance = empirical_game.variance_utils()
    print(c)
    bennett_bound = log_term_bennett * (c ** 2 / (variance * h(c * target_epsilon / variance)))
    bennett_bound = np.amax(bennett_bound, axis=-1)  # Only care about profiles
    bennett_bound = bennett_bound.reshape(-1)
    empirical_bennett_bound = 1 + beta * ((8.0 / 3 + np.sqrt(
        4 + 2 / log_term)) * c * log_term / target_epsilon + 2 * variance * log_term / target_epsilon ** 2)
    empirical_bennett_bound = np.amax(empirical_bennett_bound, axis=-1)  # Only care about profiles
    empirical_bennett_bound = empirical_bennett_bound.reshape(-1)
    var_ticks = 0.05
    max_sample_complexity = np.amax(empirical_bennett_bound)
    max_var = np.amax(variance)
    print(max_var)
    if max_var % var_ticks != 0:
        max_var += (var_ticks - max_var % var_ticks)

    sample_history = []
    proportion_pruned = []

    for n in range(num_iterations):
        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=beta,
            well_estimated_pruning=True,
            regret_pruning=False,
            show_graphs_every=-1)

        num_active_profiles = psp_results['num_active_profiles'][1:]
        proportion_pruned.extend(
            [1 - active_profiles / num_profiles for active_profiles in num_active_profiles])
        sample_history.extend(psp_sample_history)
        print(f'N: {n}')

    if ax is not None:
        ax.set_ylabel('Proportion of Active Profiles')
        bennett_ecdf = ECDF(bennett_bound)
        e_bennett_ecdf = ECDF(empirical_bennett_bound)
        bennett_ecdf.y = [1 - y for y in bennett_ecdf.y]
        proportion_pruned = [1 - y for y in proportion_pruned]
        e_bennett_ecdf.y = [1 - y for y in e_bennett_ecdf.y]
        bennett_ecdf.x[0] = 0
        e_bennett_ecdf.x[0] = 0
        ax.fill_between(e_bennett_ecdf.x, e_bennett_ecdf.y, color='mistyrose', step='post')
        ax.fill_between([0, *sample_history], [1, *proportion_pruned], color='palegreen', step='post')
        ax.fill_between(bennett_ecdf.x, bennett_ecdf.y, color='lightblue', step='post')
        sns.lineplot([0, *bennett_ecdf.x], [1, *bennett_ecdf.y], label='Bennett (Known Variance)', ax=ax,
                     color='royalblue', drawstyle='steps-post')
        sns.lineplot([0, *sample_history], [1, *proportion_pruned], color='darkgreen', alpha=0.8,
                     drawstyle='steps-post', ax=ax)
        sns.scatterplot(sample_history, [1, *proportion_pruned[:-1]], color='darkgreen', label='PSP', ax=ax)
        sns.lineplot([0, *e_bennett_ecdf.x], [1, *e_bennett_ecdf.y], label='PSP Upper Bound', ax=ax,
                     color='tomato', drawstyle='steps-post')
        ax.set_xlim(0, max_sample_complexity)
        ax.set_ylim(0, 1)
        n = int(np.floor(np.log10(ax.get_xticks()[1])))
        print(ax.get_xticks())
        print(ax.get_xticklabels())
        ax.set_xticklabels([f'{x / (10 ** n):.1f}' for x in ax.get_xticks()])
        ax.set_xlabel(f'Number of Samples ($\\cdot 10^{n}$)')
        print(ax.get_xticks())
        print(ax.get_xticklabels())
        if title is not None:
            ax.set_title(title)
    else:
        plt.figure()
        plt.ylabel('Proportion Pruned')

        plt.subplot(111)
        sns.ecdfplot(bennett_bound)
        sns.scatterplot(sample_history, proportion_pruned)
        sns.ecdfplot(empirical_bennett_bound)
        plt.legend(
            ['Bennett (Known Variance)', 'PSP', 'PSP Upper Bound'])
        plt.xlim(0, max_sample_complexity)
        plt.xlabel('Data Complexity')
        if title is not None:
            plt.title(title)
        plt.show()

    # print(len(psp_sample_history))
    # print(len(proportion_pruned))
    #
    # ax.twiny()
    # sns.lineplot(psp_sample_history, proportion_pruned)
    # plt.xlim(start_x, start_x + max_var * multiplier)
    # plt.xlabel('Sample Complexity')
    # ax2.spines['top'].set_position(('axes', -0.15))
    # ax2.spines['top'].set_visible(False)
    # plt.tick_params(which='both', top=False)


def proportion_remaining_vs_samples_subplots():
    n_rows = 2
    n_cols = 2
    # figsize = (14.5, 4.15)
    figsize = (7.25, 7.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    alphas = [0.5, 1.5, 3, 5]
    betas = [3, 3, 1.5, 0.5]
    titles = [
        'Beta(0.5, 3); Mostly low variance',
        'Beta(1.5, 3); Mostly medium variance',
        'Beta(3, 1.5); Mostly high variance',
        'Beta(5, 0.5); Almost all high variance'
    ]
    max_x_lim = 0
    for i in range(4):
        row = i // n_cols
        col = i % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        proportion_remaining_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
                                                                 complete_noise=10,
                                                                 noise_distribution=scipy.stats.bernoulli,
                                                                 noise_args={'p': 0.5},
                                                                 noise_multiplier_distribution=scipy.stats.beta,
                                                                 noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                                        'a': alphas[i],
                                                                                        'b': betas[i]}),
                                        target_epsilon=0.2, delta=0.05, beta=1.1, title=titles[i], ax=ax)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(4):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_xlim(0, max_x_lim)
        ax.legend().set_visible(False)
        if col > 0:
            ax.set_ylabel('')
        if i == 3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.show()
    fig.savefig('proportion_remaining.pdf', bbox_inches='tight')


def proportion_remaining_vs_samples_subplots_auctions():
    n_rows = 2
    n_cols = 2
    # figsize = (14.5, 4.15)
    figsize = (7.25, 7.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    players = [2, 4, 2, 4]
    a_types = [2, 2, 1, 1]
    titles = [
        '2 Player Second-Price Auction',
        '4 Player Second-Price Auction',
        '2 Player First-Price Auction',
        '4 Player First-Price Auction'
    ]
    max_x_lim = 0
    for i in range(4):
        row = i // n_cols
        col = i % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        proportion_remaining_vs_samples(
            BiddingGame(players[i], 1, 127, shading_factors=[0.3, 0.5, 0.7], auction_type=a_types[i]),
            target_epsilon=10, delta=0.05, beta=1.1, title=titles[i], ax=ax)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(4):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_xlim(0, max_x_lim)
        ax.legend().set_visible(False)
        if col > 0:
            ax.set_ylabel('')
        if i == 3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.show()
    fig.savefig('proportion_remaining_auction.pdf', bbox_inches='tight')


def empirical_bennett_vs_simplifying_inequalities(empirical_game: NoisyGame, batch_size):
    c = empirical_game.get_c()
    delta = 0.05
    game_size = np.prod(empirical_game.get_utils_shape())
    log_term = np.log(3) + np.log(game_size) - np.log(delta)
    variance = empirical_game.variance_utils()

    sample_sizes = []

    df_variance = None
    df = None

    streaming_object = None
    for i in range(1000):
        m = batch_size * (i + 1)
        sample_sizes.append(m)
        sample_utils = np.array(
            [empirical_game.sample_utils() for _ in range(batch_size)])
        stats, streaming_object = EGTA.get_stats(sample_utils, streaming=True, streaming_object=streaming_object)

        empirical_variances = stats['sample_variance'].reshape(-1)
        variance_bound = (variance + (2 * m + 1) / (6 * (m - 1)) * c ** 2 * log_term / m + np.sqrt(
            2 * c ** 2 * variance * log_term / (m - 1))).reshape(-1)
        df_variance = create_pandas_dataframe(df_variance, m, empirical_variances, variance_bound,
                                              'Sample Size', 'Variance', 'Type', 'Empirical Variance', 'Variance Bound')

        epsilon = EGTA.bennet_non_uniform(c, delta, stats).reshape(-1)
        epsilon_simplified = (4 * c * log_term / (3 * m) + np.sqrt(
            (1 / 3 + 1 / (2 * log_term) * (1 + 1 / (m - 1)) ** 2) * (c * log_term / m) ** 2 + 2 * stats[
                'sample_variance'] * log_term / m)).reshape(-1)
        epsilon_double_simplified = ((4 / 3 + np.sqrt(1 + 1 / (2 * log_term) * (1 + 1 / (m - 1)))) * c * log_term / (
                m - 1) + np.sqrt(2 * variance / (m - 1))).reshape(-1)
        # epsilon_double_simplified = (4 * c * log_term / (3 * m) + np.sqrt(
        #             (1 / 3 + 1 / (2 * log_term) * (1 + 1 / (m - 1)) ** 2) * (c * log_term / m) ** 2 + 2 * (
        #                     variance + (2 * m + 1) / (6 * (m - 1)) * c ** 2 * log_term / m + np.sqrt(
        #                 2 * c ** 2 * variance * log_term / (m - 1))) * log_term / m)).reshape(-1)

        df = create_pandas_dataframe(df, m, np.divide(epsilon_simplified, epsilon),
                                     np.divide(epsilon_double_simplified, epsilon),
                                     'Sample Size', 'Ratio', 'Type', 'Empirical Simplification',
                                     'Non-Empirical Simplification')

        print(i + 1)
        if (i + 1) % 1 == 0:
            # sns.violinplot(x='Sample Size', y='Ratio', hue='Type', data=df, split=True, inner=None, scale='count')
            # plt.show()
            sns.violinplot(x='Sample Size', y='Variance', hue='Type', data=df_variance, split=True, inner=None,
                           scale='count')
            plt.show()


# Copied from https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s = r'%s{\times}%s' % (significand, exponent)
        else:
            s = r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def eps_vs_sample_complexities(empirical_game: EmpiricalGame, delta=0.05, beta=1.1, batch_size=25,
                               num_algo_iterations=10, axes=None, show_error_bars=False):
    c = float(empirical_game.get_c())
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players
    variance = empirical_game.variance_utils()
    wimpy_variance = np.amax(variance)

    epsilons = [c / n for n in range(2, 40)]

    gs_h_data_complexities = []
    gs_b_data_complexities = []
    gs_eb_data_complexities = []
    ps_data_complexities = []
    gs_empirical_b_data_complexities = []
    psp_data_complexities = []

    gs_h_complexities = []
    gs_b_complexities = []
    gs_empirical_b_complexities = []
    gs_empirical_b_inverse_epsilons = []
    gs_eb_complexities = []
    ps_complexities = []
    psp_complexities = []
    psp_inverse_epsilons = []

    for eps in epsilons:
        gs_h_data_complexities.append(gs_h_sample_complexity(c, delta, eps, game_size))
        gs_h_complexities.append(gs_h_data_complexities[-1] * num_profiles)
        gs_b_data_complexities.append(gs_b_sample_complexity(c, delta, eps, wimpy_variance, game_size))
        gs_b_complexities.append(gs_b_data_complexities[-1] * num_profiles)
        gs_eb_data_complexities.append(gs_eb_sample_complexity(c, delta, eps, wimpy_variance, game_size))
        gs_eb_complexities.append(gs_eb_data_complexities[-1] * num_profiles)
        ps_data_complexities.append(ps_sample_complexity(c, delta, eps, wimpy_variance, beta, game_size))
        ps_complexities.append(ps_query_complexity(c, delta, eps, variance, beta, game_size))

    for n in range(num_algo_iterations):
        num_batches = 1
        min_target_eps = min(epsilons)
        while True:
            sample_history, results = global_sampling(empirical_game, c, delta, 0, batch_size,
                                                      max_iterations=num_batches,
                                                      show_graphs_every=-1, use_hoeffding=False, verbose=0)
            gs_empirical_b_data_complexities.append(batch_size * num_batches)
            gs_empirical_b_complexities.append(gs_empirical_b_data_complexities[-1] * num_profiles)
            sup_eps = results['supremum_epsilon'][-1]
            gs_empirical_b_inverse_epsilons.append(1 / sup_eps)
            print(sup_eps)
            if sup_eps <= min_target_eps:
                break
            num_batches += 1

        for eps in epsilons:
            print(eps)
            T = np.ceil(np.log(3 * c / (4 * eps)) / np.log(beta)).astype('int')
            if T <= 0:
                continue
            psp_sample_history, psp_results = progressive_sampling_with_pruning(empirical_game, c, delta, eps,
                                                                                beta,
                                                                                regret_pruning=False,
                                                                                verbose=0)
            num_active_profiles = psp_results['num_active_profiles']
            queries = psp_sample_history[0] * num_active_profiles[0]
            for i in range(1, len(psp_sample_history)):
                queries += (psp_sample_history[i] - psp_sample_history[i - 1]) * num_active_profiles[i]
            psp_data_complexities.append(psp_sample_history[-1])
            psp_complexities.append(queries)
            # psp_inverse_epsilons.append(1 / psp_results['supremum_epsilon'][-1])
            psp_inverse_epsilons.append(1 / eps)
        print('Finished Algo Run')

    error_bars_x = []
    error_bars_y = []
    error_bars = []
    if show_error_bars:
        for game_size in set(gs_empirical_b_complexities):
            inverse_epsilons = [gs_empirical_b_inverse_epsilons[i] for i in range(len(gs_empirical_b_complexities)) if
                                gs_empirical_b_complexities[i] == game_size]
            print(len(inverse_epsilons))
            error_bars_y.append(game_size)
            if len(inverse_epsilons) == 1:
                error_bars.append(0)
                error_bars_x.append(inverse_epsilons[0])
                continue
            ci = (min(inverse_epsilons), max(inverse_epsilons))
            error_bars.append((ci[1] - ci[0]) / 2.0)
            error_bars_x.append((ci[1] + ci[0]) / 2.0)

    inverse_epsilons = [1 / eps for eps in epsilons]

    psp_error_bars_x = []
    psp_error_bars_y = []
    psp_error_bars = []
    if show_error_bars:
        print(set(psp_inverse_epsilons))
        for inv_eps in set(psp_inverse_epsilons):
            complexities = [psp_complexities[i] for i in range(len(psp_inverse_epsilons)) if
                            psp_inverse_epsilons[i] == inv_eps]
            psp_error_bars_x.append(inv_eps)
            if len(complexities) == 1:
                psp_error_bars.append(0)
                psp_error_bars_y.append(complexities[0])
                continue
            ci = (min(complexities), max(complexities))
            psp_error_bars.append((ci[1] - ci[0]) / 2.0)
            psp_error_bars_y.append((ci[1] + ci[0]) / 2.0)

    error_bars_x_data = []
    error_bars_y_data = []
    error_bars_data = []
    if show_error_bars:
        for game_size in set(gs_empirical_b_data_complexities):
            inverse_epsilons = [gs_empirical_b_inverse_epsilons[i] for i in range(len(gs_empirical_b_data_complexities))
                                if gs_empirical_b_data_complexities[i] == game_size]
            error_bars_y_data.append(game_size)
            if len(inverse_epsilons) == 1:
                error_bars_data.append(0)
                error_bars_x_data.append(inverse_epsilons[0])
                continue
            ci = (min(inverse_epsilons), max(inverse_epsilons))
            error_bars_data.append((ci[1] - ci[0]) / 2.0)
            error_bars_x_data.append((ci[1] + ci[0]) / 2.0)

    inverse_epsilons = [1 / eps for eps in epsilons]

    psp_error_bars_x_data = []
    psp_error_bars_y_data = []
    psp_error_bars_data = []
    if show_error_bars:
        print(set(psp_inverse_epsilons))
        for inv_eps in set(psp_inverse_epsilons):
            complexities = [psp_data_complexities[i] for i in range(len(psp_inverse_epsilons)) if
                            psp_inverse_epsilons[i] == inv_eps]
            psp_error_bars_x_data.append(inv_eps)
            if len(complexities) == 1:
                psp_error_bars_data.append(0)
                psp_error_bars_y_data.append(complexities[0])
                continue
            ci = (min(complexities), max(complexities))
            psp_error_bars_data.append((ci[1] - ci[0]) / 2.0)
            psp_error_bars_y_data.append((ci[1] + ci[0]) / 2.0)

    max_y = max(max(gs_eb_complexities), max(ps_complexities))

    axes_was_none = False
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes_was_none = True

    fmt = 's'
    markersize = 3

    sns.lineplot(inverse_epsilons, gs_h_complexities, ax=axes[1], color='green', label='GS-H')
    sns.lineplot(inverse_epsilons, gs_b_complexities, ax=axes[1], color='red', label='GS-B (Known Variance)')
    if show_error_bars:
        axes[1].errorbar(error_bars_x, error_bars_y, xerr=error_bars, yerr=None, fmt=fmt, markersize=markersize,
                         ls='none', color='royalblue',
                         label='GS-EB')
    else:
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_complexities, ax=axes[1], color='royalblue',
                        label='GS-EB')
    sns.lineplot(inverse_epsilons, gs_eb_complexities, ax=axes[1], color='royalblue', label='GS-EB Upper Bound')
    if show_error_bars:
        axes[1].errorbar(psp_error_bars_x, psp_error_bars_y, xerr=None, yerr=psp_error_bars, fmt=fmt,
                         markersize=markersize, ls='none',
                         color='darkorange', label='PSP')
    else:
        sns.scatterplot(psp_inverse_epsilons, psp_complexities, ax=axes[1], color='darkorange', label='PSP')
    sns.lineplot(inverse_epsilons, ps_complexities, ax=axes[1], color='darkorange', label='PSP Upper Bound')
    axes[1].set_xlabel('1/$\\varepsilon$')
    axes[1].set_ylim(0, max_y)
    n = int(np.floor(np.log10(axes[1].get_yticks()[1])))
    axes[1].set_yticklabels([f'{y / (10 ** n):.1f}' for y in axes[1].get_yticks()])
    axes[1].set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    axes[1].set_title('Query Complexity vs $1/\\varepsilon$ (Single Game)')

    max_y = max(max(gs_eb_data_complexities), max(ps_data_complexities))

    sns.lineplot(inverse_epsilons, gs_h_data_complexities, ax=axes[0], color='green', label='GS-H')
    sns.lineplot(inverse_epsilons, gs_b_data_complexities, ax=axes[0], color='red', label='GS-B (Known Variance)')
    if show_error_bars:
        axes[0].errorbar(error_bars_x_data, error_bars_y_data, xerr=error_bars_data, yerr=None, fmt=fmt,
                         markersize=markersize, ls='none', color='royalblue',
                         label='GS-EB')
    else:
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_data_complexities, ax=axes[0],
                        color='royalblue',
                        label='GS-EB')
    sns.lineplot(inverse_epsilons, gs_eb_data_complexities, ax=axes[0], color='royalblue', label='GS-EB Upper Bound')
    if show_error_bars:
        axes[0].errorbar(psp_error_bars_x_data, psp_error_bars_y_data, xerr=None, yerr=psp_error_bars_data, fmt=fmt,
                         markersize=markersize, ls='none',
                         color='darkorange', label='PSP')
    else:
        sns.scatterplot(psp_inverse_epsilons, psp_data_complexities, ax=axes[0], color='darkorange', label='PSP')
    sns.lineplot(inverse_epsilons, ps_data_complexities, ax=axes[0], color='darkorange', label='PSP Upper Bound')
    axes[0].set_xlabel('1/$\\varepsilon$')
    axes[0].set_ylim(0, max_y)
    n = int(np.floor(np.log10(axes[0].get_yticks()[1])))
    axes[0].set_yticklabels([f'{y / (10 ** n):.1f}' for y in axes[0].get_yticks()])
    axes[0].set_ylabel(f'Data Complexity ($\\cdot 10^{n}$)')
    axes[0].set_title('Data Complexity vs $1/\\varepsilon$ (Single Game)')

    if axes_was_none:
        plt.tight_layout()
        plt.show()


def eps_vs_sample_complexities_many_games(empirical_games: List[NoisyGame], delta=0.05, beta=1.1, batch_size=25,
                                          ax=None, show_error_bars=False):
    c = float(empirical_games[0].get_c())
    game_size = np.prod(empirical_games[0].get_utils_shape())
    num_players = empirical_games[0].get_utils_shape()[-1]
    num_profiles = game_size / num_players
    variances = [empirical_game.variance_utils() for empirical_game in empirical_games]
    wimpy_variances = [np.amax(variance) for variance in variances]

    epsilons = [c / n for n in range(2, 40)]

    bound_inverse_epsilons = []
    gs_h_complexities = []
    gs_b_complexities = []
    gs_empirical_b_complexities = []
    gs_empirical_b_inverse_epsilons = []
    gs_eb_complexities = []
    ps_complexities = []
    psp_complexities = []
    psp_inverse_epsilons = []

    for i in range(len(empirical_games)):
        for eps in epsilons:
            bound_inverse_epsilons.append(1 / eps)
            gs_h_complexities.append(num_profiles * gs_h_sample_complexity(c, delta, eps, game_size))
            gs_b_complexities.append(
                num_profiles * gs_b_sample_complexity(c, delta, eps, wimpy_variances[i], game_size))
            gs_eb_complexities.append(
                num_profiles * gs_eb_sample_complexity(c, delta, eps, wimpy_variances[i], game_size))
            ps_complexities.append(ps_query_complexity(c, delta, eps, variances[i], beta, game_size))

    for i in range(len(empirical_games)):
        print(i)
        num_batches = 1
        min_target_eps = min(epsilons)
        while True:
            sample_history, results = global_sampling(empirical_games[i], c, delta, 0, batch_size,
                                                      max_iterations=num_batches,
                                                      show_graphs_every=-1, use_hoeffding=False, verbose=0)
            gs_empirical_b_complexities.append(batch_size * num_batches * num_profiles)
            sup_eps = results['supremum_epsilon'][-1]
            gs_empirical_b_inverse_epsilons.append(1 / sup_eps)
            if sup_eps <= min_target_eps:
                break
            num_batches += 1

        for eps in epsilons:
            print(eps)
            T = schedule_length(c, eps, beta)
            if T <= 0:
                continue
            psp_sample_history, psp_results = progressive_sampling_with_pruning(empirical_games[i], c, delta, eps,
                                                                                beta,
                                                                                regret_pruning=False,
                                                                                verbose=0)
            num_active_profiles = psp_results['num_active_profiles']
            queries = psp_sample_history[0] * num_active_profiles[0]
            for j in range(1, len(psp_sample_history)):
                queries += (psp_sample_history[j] - psp_sample_history[j - 1]) * num_active_profiles[j]
            psp_complexities.append(queries)
            # psp_inverse_epsilons.append(1 / psp_results['supremum_epsilon'][-1])
            psp_inverse_epsilons.append(1 / eps)
        print('Finished Algo Run')

    error_bars_x = []
    error_bars_y = []
    error_bars = []
    if show_error_bars:
        for complexity in set(gs_empirical_b_complexities):
            inverse_epsilons = [gs_empirical_b_inverse_epsilons[i] for i in range(len(gs_empirical_b_complexities)) if
                                gs_empirical_b_complexities[i] == complexity]
            print(len(inverse_epsilons))
            if len(inverse_epsilons) < len(empirical_games):
                continue
            error_bars_y.append(complexity)
            if len(inverse_epsilons) == 1:
                error_bars.append(0)
                error_bars_x.append(inverse_epsilons[0])
                continue
            ci = (min(inverse_epsilons), max(inverse_epsilons))
            error_bars.append((ci[1] - ci[0]) / 2.0)
            error_bars_x.append((ci[1] + ci[0]) / 2.0)

    psp_error_bars_x = []
    psp_error_bars_y = []
    psp_error_bars = []
    if show_error_bars:
        for inv_eps in set(psp_inverse_epsilons):
            complexities = [psp_complexities[i] for i in range(len(psp_inverse_epsilons)) if
                            psp_inverse_epsilons[i] == inv_eps]
            psp_error_bars_x.append(inv_eps)
            if len(complexities) == 1:
                psp_error_bars.append(0)
                psp_error_bars_y.append(complexities[0])
                continue
            ci = (min(complexities), max(complexities))
            psp_error_bars.append((ci[1] - ci[0]) / 2.0)
            psp_error_bars_y.append((ci[1] + ci[0]) / 2.0)

    max_y = max(max(gs_eb_complexities), max(ps_complexities))

    ax_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax_was_none = True

    fmt = 's'
    markersize = 3

    sns.lineplot(bound_inverse_epsilons, gs_h_complexities, ax=ax, color='green', label='GS-H')
    sns.lineplot(bound_inverse_epsilons, gs_b_complexities, ax=ax, color='red', label='GS-B (Known Variance)')
    if show_error_bars:
        ax.errorbar(error_bars_x, error_bars_y, xerr=error_bars, yerr=None, fmt=fmt, markersize=markersize, ls='none',
                    color='royalblue',
                    label='GS-EB')
    else:
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_complexities, ax=ax, color='royalblue',
                        label='GS-EB')
    sns.lineplot(bound_inverse_epsilons, gs_eb_complexities, ax=ax, color='royalblue', label='GS-EB Upper Bound')
    if show_error_bars:
        ax.errorbar(psp_error_bars_x, psp_error_bars_y, xerr=None, yerr=psp_error_bars, fmt=fmt, markersize=markersize,
                    ls='none',
                    color='darkorange', label='PSP')
    else:
        sns.scatterplot(psp_inverse_epsilons, psp_complexities, ax=ax, color='darkorange', label='PSP')
    sns.lineplot(bound_inverse_epsilons, ps_complexities, ax=ax, color='darkorange', label='PSP Upper Bound')
    ax.set_xlabel('1/$\\varepsilon$')
    ax.set_ylim(0, max_y)
    n = int(np.floor(np.log10(ax.get_yticks()[1])))
    ax.set_yticklabels([f'{y / (10 ** n):.1f}' for y in ax.get_yticks()])
    ax.set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    ax.set_title('Query Complexity vs $1/\\varepsilon$ (Many Games)')

    if ax_was_none:
        plt.tight_layout()
        plt.show()


def eps_vs_sample_complexities_final_subplots():
    shift = 0.05
    alpha = 1.5
    beta = 3
    shifts = list(itertools.product([-2 * shift, -1 * shift, 0, shift, 2 * shift],
                                    [-2 * shift, -1 * shift, 0, shift, 2 * shift]))

    fig, axes = plt.subplots(1, 3, figsize=(16 * 0.9, 5 * 0.9))

    eps_vs_sample_complexities_many_games([construct_empirical_game(congestion_game(3, 3, c=2),
                                                                    complete_noise=10,
                                                                    noise_distribution=scipy.stats.bernoulli,
                                                                    noise_args={'p': 0.5},
                                                                    noise_multiplier_distribution=scipy.stats.beta,
                                                                    noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                                           'a': alpha + shifts[i][0],
                                                                                           'b': beta + shifts[i][1]})
                                           for i in
                                           range(len(shifts))],
                                          batch_size=200, ax=axes[2], show_error_bars=True)
    eps_vs_sample_complexities(construct_empirical_game(congestion_game(3, 3, c=2),
                                                        complete_noise=10,
                                                        noise_distribution=scipy.stats.bernoulli,
                                                        noise_args={'p': 0.5},
                                                        noise_multiplier_distribution=scipy.stats.beta,
                                                        noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                               'a': alpha,
                                                                               'b': beta}),
                               batch_size=200, num_algo_iterations=25, axes=axes[:2], show_error_bars=True)
    plt.tight_layout()
    plt.show()
    fig.savefig('eps_vs_sample_complexities.pdf', bbox_inches='tight')


def eps_vs_sample_complexities_final_subplots_auctions():
    n_cols = 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 5 * n_rows))

    eps_vs_sample_complexities(BiddingGame(3, 4, 127, [0, 0.2, 0.4, 0.6, 0.8, 1], auction_type=1),
                               batch_size=200, num_algo_iterations=1, axes=axes[0, :2], show_error_bars=False)
    eps_vs_sample_complexities(BiddingGame(3, 4, 127, [0, 0.2, 0.4, 0.6, 0.8, 1], auction_type=2),
                               batch_size=200, num_algo_iterations=1, axes=axes[1, :2], show_error_bars=False)

    axes[0, 0].set_title('First-Price; Data Complexity')
    axes[0, 1].set_title('First-Price; Query Complexity')
    axes[1, 0].set_title('Second-Price; Data Complexity')
    axes[1, 1].set_title('Second-Price; Query Complexity')

    plt.tight_layout()
    plt.show()
    fig.savefig('eps_vs_sample_complexities_auctions.pdf', bbox_inches='tight')


def eps_vs_sample_complexities_amy(delta=0.05,
                                   num_algo_iterations=1):
    N = 4
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5))
    batch_sizes = [25, 100, 200, 300]

    c = 22

    for var_idx in range(N):
        empirical_game = construct_empirical_game(random_zero_sum(actions=20, c=2),
                                                  complete_noise=10 * np.sqrt(var_idx / (N - 1)),
                                                  noise_distribution=scipy.stats.bernoulli,
                                                  noise_args={'p': 0.5})

        game_size = np.prod(empirical_game.get_utils_shape())
        variance = empirical_game.variance_utils()
        wimpy_variance = np.amax(variance)

        epsilons = [c / n for n in range(2, 40)]

        gs_h_data_complexities = []
        gs_b_data_complexities = []
        gs_eb_data_complexities = []
        gs_empirical_b_data_complexities = []
        gs_empirical_b_inverse_epsilons = []

        for eps in epsilons:
            gs_h_data_complexities.append(gs_h_sample_complexity(c, delta, eps, game_size))
            gs_b_data_complexities.append(gs_b_sample_complexity(c, delta, eps, wimpy_variance, game_size))
            gs_eb_data_complexities.append(gs_eb_sample_complexity(c, delta, eps, wimpy_variance, game_size))

        for n in range(num_algo_iterations):
            batch_size = batch_sizes[var_idx]
            num_batches = 1
            min_target_eps = min(epsilons)
            print(min_target_eps)
            while True:
                sample_history, results = global_sampling(empirical_game, c, delta, 0, batch_size,
                                                          max_iterations=num_batches,
                                                          show_graphs_every=-1, use_hoeffding=False, verbose=0)
                gs_empirical_b_data_complexities.append(batch_size * num_batches)
                sup_eps = results['supremum_epsilon'][-1]
                print(sup_eps)
                gs_empirical_b_inverse_epsilons.append(1 / sup_eps)
                if sup_eps <= min_target_eps:
                    break
                num_batches += 1

            print('Finished Algo Run')

        inverse_epsilons = [1 / eps for eps in epsilons]

        max_y = max(gs_eb_data_complexities)

        sns.lineplot(inverse_epsilons, gs_h_data_complexities, ax=axes[var_idx], color='green', label='GS-H')
        sns.lineplot(inverse_epsilons, gs_b_data_complexities, ax=axes[var_idx], color='red',
                     label='GS-B (Known Variance)')
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_data_complexities, ax=axes[var_idx],
                        color='royalblue',
                        label='GS-EB')
        sns.lineplot(inverse_epsilons, gs_eb_data_complexities, ax=axes[var_idx], color='royalblue',
                     label='GS-EB Upper Bound')
        axes[var_idx].set_xlabel('1/$\\varepsilon$')
        axes[var_idx].set_ylabel('Sample Complexity')
        axes[var_idx].set_ylim(0, max_y)
        axes[var_idx].yaxis.set_major_formatter(MathTextSciFormatter('%1.1e'))
        axes[var_idx].set_title(f'$||v||_\\infty = {100 * var_idx / (N - 1):.1f}$; $c=22$')
    fig.suptitle('')
    plt.tight_layout()
    plt.show()
    fig.savefig('gs_wimpy_variance_greenwald.pdf', bbox_inches='tight')


def eps_vs_query_complexities_amy_bidding_game(delta=0.05, num_algo_iterations=1, num_epsilons=40, variance_m=30000,
                                               fig_scale=5):
    N = 4
    fig, axes = plt.subplots(1, N, figsize=(fig_scale * N, fig_scale * 1.1))

    alphas = [0.1, 1, 2, 2]
    betas = [2, 1, 0.5, 0.01]

    empirical_games = [BiddingGame(3, 1, 127, [0, 0.2, 0.4, 0.6, 0.8, 1], auction_type=1, random_distributions=True)
                       for _ in range(N)]
    for idx in range(N):
        for i in range(len(empirical_games[idx].alphas)):
            empirical_games[idx].alphas[i] = alphas[idx]
            empirical_games[idx].betas[i] = betas[idx]
    c = empirical_games[0].get_c()
    batch_sizes = [50, 100, 150, 200, 200]

    n = 3
    for idx in range(N):
        empirical_game = empirical_games[idx]

        game_size = np.prod(empirical_game.get_utils_shape())
        num_players = empirical_game.get_utils_shape()[-1]
        num_profiles = game_size / num_players
        variance = empirical_game.variance_utils(m=variance_m)
        wimpy_variance = np.amax(variance)

        epsilons = [c / n for n in range(2, num_epsilons)]

        gs_h_complexities = []
        gs_b_complexities = []
        gs_eb_complexities = []
        gs_empirical_b_complexities = []
        gs_empirical_b_inverse_epsilons = []

        for eps in epsilons:
            gs_h_complexities.append(gs_h_sample_complexity(c, delta, eps, game_size))
            gs_b_complexities.append(gs_b_sample_complexity(c, delta, eps, wimpy_variance, game_size))
            gs_eb_complexities.append(gs_eb_sample_complexity(c, delta, eps, wimpy_variance, game_size))

        for n in range(num_algo_iterations):
            batch_size = batch_sizes[idx]
            num_batches = 1
            min_target_eps = min(epsilons)
            print(min_target_eps)
            while True:
                sample_history, results = global_sampling(empirical_game, c, delta, 0, batch_size,
                                                          max_iterations=num_batches,
                                                          show_graphs_every=-1, use_hoeffding=False, verbose=0)
                gs_empirical_b_complexities.append(batch_size * num_batches)
                sup_eps = results['supremum_epsilon'][-1]
                print(sup_eps)
                gs_empirical_b_inverse_epsilons.append(1 / sup_eps)
                if sup_eps <= min_target_eps:
                    break
                num_batches += 1

            print('Finished Algo Run')

        inverse_epsilons = [1 / eps for eps in epsilons]

        max_y = max(gs_eb_complexities)
        if idx == 0:
            n = int(math.floor(np.log10(max_y)))

        gs_h_complexities = [complexity / (10 ** n) for complexity in gs_h_complexities]
        gs_b_complexities = [complexity / (10 ** n) for complexity in gs_b_complexities]
        gs_eb_complexities = [complexity / (10 ** n) for complexity in gs_eb_complexities]
        gs_empirical_b_complexities = [complexity / (10 ** n) for complexity in gs_empirical_b_complexities]

        sns.lineplot(inverse_epsilons, gs_h_complexities, ax=axes[idx], color='green', label='GS-H')
        sns.lineplot(inverse_epsilons, gs_b_complexities, ax=axes[idx], color='red',
                     label='GS-B (Known Variance)')
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_complexities, ax=axes[idx],
                        color='royalblue',
                        label='GS-EB')
        sns.lineplot(inverse_epsilons, gs_eb_complexities, ax=axes[idx], color='royalblue',
                     label='GS-EB Upper Bound')
        axes[idx].set_xlabel('1/$\\varepsilon$')
        if idx == 0:
            axes[idx].set_ylabel(f'Sample Complexity ($\\cdot 10^{n}$)')
        axes[idx].set_ylim(0, max_y / 10 ** n)
        axes[idx].set_title(
            f'$(\\alpha, \\beta) = ({alphas[idx]}, {betas[idx]})$; $||v||_\\infty \\approx {wimpy_variance:.1f}$')

    fig.suptitle('Global Sampling: Hoeffding vs Empirical Bennett')
    plt.tight_layout()
    plt.show()
    fig.savefig('gs_with_varying_variances_bidding_game.pdf', bbox_inches='tight')


def eps_vs_query_complexities_amy(delta=0.05, num_algo_iterations=1, beta=1.1, bidding_game_version=False):
    N = 4
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5))

    batch_sizes = [25, 100, 200, 300]
    c = 22
    empirical_games = [construct_empirical_game(random_zero_sum(actions=30, c=2),
                                                complete_noise=10 * np.sqrt(var_idx / (N - 1)),
                                                noise_distribution=scipy.stats.bernoulli,
                                                noise_args={'p': 0.5}) for var_idx in range(N)]

    for idx in range(N):
        empirical_game = empirical_games[idx]

        game_size = np.prod(empirical_game.get_utils_shape())
        num_players = empirical_game.get_utils_shape()[-1]
        num_profiles = game_size / num_players
        variance = empirical_game.variance_utils()
        wimpy_variance = np.amax(variance)

        epsilons = [c / n for n in range(2, 40)]

        gs_h_complexities = []
        gs_b_complexities = []
        gs_eb_complexities = []
        gs_empirical_b_complexities = []
        gs_empirical_b_inverse_epsilons = []
        psp_complexities = []
        psp_inverse_epsilons = []

        for eps in epsilons:
            gs_h_complexities.append(num_profiles * gs_h_sample_complexity(c, delta, eps, game_size))
            gs_b_complexities.append(num_profiles * gs_b_sample_complexity(c, delta, eps, wimpy_variance, game_size))
            gs_eb_complexities.append(num_profiles * gs_eb_sample_complexity(c, delta, eps, wimpy_variance, game_size))

        for n in range(num_algo_iterations):
            batch_size = batch_sizes[idx]
            num_batches = 1
            min_target_eps = min(epsilons)
            print(min_target_eps)
            while True:
                sample_history, results = global_sampling(empirical_game, c, delta, 0, batch_size,
                                                          max_iterations=num_batches,
                                                          show_graphs_every=-1, use_hoeffding=False, verbose=0)
                gs_empirical_b_complexities.append(num_profiles * batch_size * num_batches)
                sup_eps = results['supremum_epsilon'][-1]
                print(sup_eps)
                gs_empirical_b_inverse_epsilons.append(1 / sup_eps)
                if sup_eps <= min_target_eps:
                    break
                num_batches += 1

            for eps in epsilons:
                T = schedule_length(c, eps, beta)
                if T <= 0:
                    continue
                psp_sample_history, psp_results = progressive_sampling_with_pruning(empirical_game, c, delta, eps,
                                                                                    beta,
                                                                                    well_estimated_pruning=False,
                                                                                    regret_pruning=True,
                                                                                    verbose=2)
                num_active_profiles = psp_results['num_active_profiles']
                queries = psp_sample_history[0] * num_active_profiles[0]
                for j in range(1, len(psp_sample_history)):
                    queries += (psp_sample_history[j] - psp_sample_history[j - 1]) * num_active_profiles[j]
                psp_complexities.append(queries)
                # psp_inverse_epsilons.append(1 / psp_results['supremum_epsilon'][-1])
                psp_inverse_epsilons.append(1 / eps)

            print('Finished Algo Run')

        inverse_epsilons = [1 / eps for eps in epsilons]

        max_y = max(gs_eb_complexities)

        sns.lineplot(inverse_epsilons, gs_h_complexities, ax=axes[idx], color='green', label='GS-H')
        sns.lineplot(inverse_epsilons, gs_b_complexities, ax=axes[idx], color='red',
                     label='GS-B (Known Variance)')
        sns.scatterplot(gs_empirical_b_inverse_epsilons, gs_empirical_b_complexities, ax=axes[idx],
                        color='royalblue',
                        label='GS-EB')
        sns.lineplot(inverse_epsilons, gs_eb_complexities, ax=axes[idx], color='royalblue',
                     label='GS-EB Upper Bound')
        sns.scatterplot(psp_inverse_epsilons, psp_complexities, ax=axes[idx], color='darkorange',
                        label='PS w/ Regret Pruning')
        axes[idx].set_xlabel('1/$\\varepsilon$')
        axes[idx].set_ylabel('Query Complexity')
        axes[idx].set_ylim(0, max_y)
        axes[idx].yaxis.set_major_formatter(MathTextSciFormatter('%1.1e'))
        axes[idx].set_title(f'$||v||_\\infty = {100 * idx / (N - 1):.1f}$; $c=22$')

    plt.tight_layout()
    plt.show()
    fig.savefig('gs_vs_regret_pruning_greenwald.pdf', bbox_inches='tight')


def generate_random_eps_game(utils, epsilon):
    return utils + scipy.stats.uniform.rvs(size=utils.shape, loc=-epsilon, scale=2 * epsilon)


def power_mean_welfare_error(game: GamutGame, eps=0.02):
    # sample_history, results = progressive_sampling_with_pruning(empirical_game, c, delta, eps, beta,
    #                                                             regret_pruning=False)
    #
    # final_utils = results['final_empirical_utilities']
    true_utils = np.maximum(0.00001, game.get_utils() + game.get_c() / 2.0)

    rhos = np.linspace(-10, 10, 100).tolist()
    x_rhos = []
    supremum_error = []
    error_supremum = []
    upper_bound = []

    for i in range(100):
        final_utils = np.maximum(0.000000001, generate_random_eps_game(true_utils, eps))
        for rho in rhos:
            x_rhos.append(rho)
            estimated_welfare_matrix = power_mean_welfare_matrix(final_utils, rho)
            true_welfare_matrix = power_mean_welfare_matrix(true_utils, rho)
            supremum_error.append(np.amax(np.abs(true_welfare_matrix - estimated_welfare_matrix)))
            error_supremum.append(np.abs(np.amin(true_welfare_matrix) - np.amin(estimated_welfare_matrix)))
            if rho < 0:
                upper_bound.append(np.power(1.0 / (len(final_utils.shape) - 1), 1 / rho) * eps)
            elif rho > 1:
                upper_bound.append(eps)
            else:
                upper_bound.append(np.nan)

    sns.lineplot(x_rhos, supremum_error, ci='sd', label='SE')
    plt.title(f'$\\rho$-Power Mean Welfare Error with $||\\Gamma - \\Gamma\'||\\leq{eps}$')
    plt.ylabel('$\\rho$-Power Mean Welfare Supremum Error (SE)')
    plt.xlabel('$\\rho$')
    sns.lineplot(x_rhos, upper_bound, ci='sd', label='Upper Bound')
    # plt.ylim(0, 0.15)
    plt.ylim(0, max(supremum_error) * 1.1)
    plt.axvline(1.0, linestyle='--', color='green', label='$\\rho=\\pm 1$')
    plt.axvline(-1.0, linestyle='--', color='green')
    plt.axhline(eps, linestyle='-.', color='green', label=f'SE={eps}')
    plt.legend()
    plt.show()

    sns.lineplot(x_rhos, error_supremum, ci='sd', label='Error')
    plt.title(f'$\\rho$-Power Mean Welfare Error with $||\\Gamma - \\Gamma\'||\\leq{eps}$')
    plt.ylabel('$Maximum \\rho$-Power Mean Welfare Error')
    plt.xlabel('$\\rho$')
    sns.lineplot(x_rhos, upper_bound, ci='sd', label='Upper Bound')
    plt.ylim(0, 0.15)
    plt.axvline(1.0, linestyle='--', color='green', label='$\\rho=\\pm 1$')
    plt.axvline(-1.0, linestyle='--', color='green')
    plt.axhline(eps, linestyle='-.', color='green', label=f'Error={eps}')
    plt.legend()
    plt.show()


def power_mean_welfare_error_sigmoidal(game: GamutGame, eps=0.02):
    true_utils = np.maximum(0.00001, game.get_utils() + game.get_c() / 2.0)

    a = 0.5
    x_rhos = np.linspace(0, 1, 1000).tolist()
    print(x_rhos)
    final_x_rhos = []
    supremum_error = []
    error_supremum = []
    upper_bound = []

    for i in range(100):
        final_utils = np.maximum(0.000000001, generate_random_eps_game(true_utils, eps))
        for x_rho in x_rhos:
            final_x_rhos.append(x_rho)
            if x_rho == 0:
                rho = -np.inf
            elif x_rho == 1:
                rho = np.inf
            else:
                rho = logit(x_rho) / a
            estimated_welfare_matrix = power_mean_welfare_matrix(final_utils, rho)
            true_welfare_matrix = power_mean_welfare_matrix(true_utils, rho)
            supremum_error.append(np.amax(np.abs(true_welfare_matrix - estimated_welfare_matrix)))
            error_supremum.append(np.abs(np.amin(true_welfare_matrix) - np.amin(estimated_welfare_matrix)))
            if x_rho == 0 or x_rho == 1:
                upper_bound.append(eps)
            elif rho < 0:
                upper_bound.append(np.minimum(np.power(1.0 / (len(final_utils.shape) - 1), 1 / rho) * eps,
                                              game.get_c()))
            elif rho > 1:
                upper_bound.append(eps)
            else:
                upper_bound.append(game.get_c())

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 10]})
    f.subplots_adjust(hspace=0.05)
    ci = 'sd'
    sns.lineplot(final_x_rhos, supremum_error, ax=ax2, ci=ci, label='SE', color='royalblue')
    f.suptitle(f'$\\rho$-Power Mean Welfare Error with $||\\Gamma - \\Gamma\'||\\leq{eps}$')
    plt.ylabel('$\\rho$-Power Mean Welfare Supremum Error (SE)')
    plt.xlabel('$\\rho$')
    sns.lineplot(final_x_rhos, upper_bound, ax=ax, ci=ci, color='tomato')
    sns.lineplot(final_x_rhos, upper_bound, ax=ax2, ci=ci, color='tomato', label='Upper Bound')
    # plt.ylim(0, 0.15)
    # plt.ylim(0, max(supremum_error) * 1.1)
    ax.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax.set_ylim(max(supremum_error) * 1.2, 2.3)
    ax.set_yticks([2], ['2'])
    ax2.set_ylim(0, max(supremum_error) * 1.1)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    COLOR = 'black'
    COLOR2 = 'black'
    ALPHA = 0.3
    ALPHA2 = 0.5
    LINESTYLE = '--'
    LINESTYLE2 = '-.'

    ax.axvline(expit(a), linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax.axvline(0.5, linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax.axvline(expit(-a), linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax2.axvline(expit(a), linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax2.axvline(0.5, linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax2.axvline(expit(-a), linestyle=LINESTYLE, color=COLOR, alpha=ALPHA)
    ax.axhline(eps, linestyle=LINESTYLE2, color=COLOR2, label=f'SE = {eps}', alpha=ALPHA2)
    ax2.axhline(eps, linestyle=LINESTYLE2, color=COLOR2, label=f'SE = {eps}', alpha=ALPHA2)

    ax2.set_xticks(
        [0, expit(-4 * a), expit(-2 * a), expit(-a), expit(-0.75 * a), expit(-0.5 * a), expit(-0.25 * a), 0.5,
         expit(0.25 * a), expit(0.5 * a), expit(0.75 * a), expit(1 * a), expit(2 * a), expit(4 * a), 1],
        ['$-\\infty$', '$-4$', '$-2$', '$-1$', '', '$-\\frac{1}{2}$', '', '$0$',
         '', '$\\frac{1}{2}$', '', '$1$', '$2$', '$4$', '$\\infty$'])

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.legend()
    plt.show()

    f.savefig('welfare.pdf', bbox_inches='tight')


def proportion_regret_pruned_vs_samples(empirical_game: NoisyGame, delta=0.05, beta=1.5, target_epsilon=5.0,
                                        use_alphas=True, num_iterations=1, title=None, ax=None):
    c = empirical_game.get_c()
    utils = empirical_game.expected_utils()
    variances = empirical_game.variance_utils()
    regret_complexities = regret_pruning_complexities(c, target_epsilon, utils, variances, delta, beta,
                                                      use_alphas)
    # regret_complexities = np.amax(regret_complexities, axis=-1).reshape(-1)
    regret_complexities = regret_complexities.reshape(-1)
    empirical_bennett_bound = ps_sample_complexity(c, delta, target_epsilon, variances, beta,
                                                   np.prod(empirical_game.get_utils_shape()))
    # empirical_bennett_bound = np.amax(empirical_bennett_bound, axis=-1).reshape(-1)
    empirical_bennett_bound = empirical_bennett_bound.reshape(-1)
    maximum_sample_complexity = np.amax(empirical_bennett_bound)
    if ax is not None:
        ax.set_ylabel('Proportion Pruned')
        sns.ecdfplot(regret_complexities, label='Regret Pruning', ax=ax)
        sns.ecdfplot(empirical_bennett_bound, label='Well-Estimated Pruning', ax=ax)
        sns.ecdfplot(np.minimum(regret_complexities, empirical_bennett_bound), label='Both', ax=ax)
        ax.set_xlim(0, maximum_sample_complexity)
        ax.set_xlabel('Sample Complexity per $(p, s)$ index')
        if title is not None:
            ax.set_title(title)
    else:
        plt.figure()
        plt.ylabel('Proportion Pruned')

        plt.subplot(111)
        sns.ecdfplot(regret_complexities, label='Regret Pruning')
        sns.ecdfplot(empirical_bennett_bound, label='Well-Estimated Pruning')
        sns.ecdfplot(np.minimum(regret_complexities, empirical_bennett_bound), label='Both')
        plt.xlim(0, maximum_sample_complexity)
        plt.legend()
        plt.show()
        plt.xlabel('Sample Complexity per Profile')
        if title is not None:
            plt.title(title)
        plt.show()


def proportion_regret_pruned_vs_samples_subplots(use_alphas=True):
    n_rows = 1
    n_cols = 4
    figsize = (16, 4.65)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    alphas = [0.5, 1.5, 1.5, 5]
    betas = [1.5, 3, 0.5, 0.5]
    titles = [
        'Beta(0.5, 1.5) — Mostly low variance',
        'Beta(3, 3) — Mostly medium variance',
        'Beta(1.5, 0.5) — Mostly high variance',
        'Beta(5, 0.5) — Almost all high variance'
    ]
    max_x_lim = 0
    for i in range(4):
        row = i // n_cols
        col = i % n_cols

        if row > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        proportion_regret_pruned_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
                                                                     complete_noise=10,
                                                                     noise_distribution=scipy.stats.bernoulli,
                                                                     noise_args={'p': 0.5},
                                                                     noise_multiplier_distribution=scipy.stats.beta,
                                                                     noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                                            'a': alphas[i],
                                                                                            'b': betas[i]}),
                                            target_epsilon=0.2, beta=1.1, use_alphas=use_alphas, title=titles[i], ax=ax)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(4):
        row = i // n_cols
        col = i % n_cols
        if row > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_xlim(0, max_x_lim)
        ax.legend().set_visible(False)
        if i == 3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center')
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.show()


def aaai_proportion_pruned_vs_samples(empirical_game: EmpiricalGame, delta=0.05, beta=1.5, target_epsilon=5.0,
                                      num_iterations=1, profiles=True, old_regret=False, title=None, ax=None,
                                      x_label_scale=0, show_y_label=False):
    c = empirical_game.get_c()
    T = schedule_length(c, target_epsilon, beta)
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players

    # log_term = np.log(2) + np.log(game_size) + np.log(T) - np.log(delta)
    # multiplier = 2 * log_term / (target_epsilon ** 2)
    # start_x = multiplier * c * target_epsilon / 3

    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    log_term_bennett = log_term - np.log(3) + np.log(2)

    # variance = empirical_game.variance_utils()
    # print(c)
    # bennett_bound = log_term_bennett * (c ** 2 / (variance * h(c * target_epsilon / variance)))
    # bennett_bound = np.amax(bennett_bound, axis=-1)  # Only care about profiles
    # bennett_bound = bennett_bound.reshape(-1)
    # empirical_bennett_bound = 1 + beta * ((8.0 / 3 + np.sqrt(
    #     4 + 2 / log_term)) * c * log_term / target_epsilon + 2 * variance * log_term / target_epsilon ** 2)
    # empirical_bennett_bound = np.amax(empirical_bennett_bound, axis=-1)  # Only care about profiles
    # empirical_bennett_bound = empirical_bennett_bound.reshape(-1)
    # var_ticks = 0.05
    # max_sample_complexity = np.amax(empirical_bennett_bound)
    # max_var = np.amax(variance)
    # print(max_var)
    # if max_var % var_ticks != 0:
    #     max_var += (var_ticks - max_var % var_ticks)

    sample_history = []
    proportion_pruned = []
    regret_history = []
    regret_pruned = []
    well_estimated_history = []
    well_estimated_pruned = []

    print(f'NUM_ITERATIONS: {T}')
    for n in range(num_iterations):
        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=beta,
            well_estimated_pruning=True,
            regret_pruning=True,
            old_regret_pruning=old_regret,
            count_each_pruning_contribution=True,
            show_graphs_every=-1,
            verbose=2)

        if profiles:
            num_active_profiles = psp_results['num_active_profiles'][1:]
            reg_active_profiles = psp_results['regret_active_profiles'][1:]
            we_active_profiles = psp_results['well_estimated_active_profiles'][1:]
            prop_pruned = [1 - active_profiles / num_profiles for active_profiles in num_active_profiles]
            reg_prop_pruned = [1 - active_profiles / num_profiles for active_profiles in reg_active_profiles]
            we_prop_pruned = [1 - active_profiles / num_profiles for active_profiles in we_active_profiles]
        else:
            num_active_indices = psp_results['num_active_utils'][1:]
            reg_active_indices = psp_results['regret_active_utils'][1:]
            we_active_indices = psp_results['well_estimated_active_utils'][1:]
            prop_pruned = [1 - active_indices / game_size for active_indices in num_active_indices]
            reg_prop_pruned = [1 - active_indices / game_size for active_indices in reg_active_indices]
            we_prop_pruned = [1 - active_indices / game_size for active_indices in we_active_indices]

        for i in range(len(psp_sample_history)):
            psp_sample_history[i] /= 10 ** x_label_scale

        well_estimated_history.extend(psp_sample_history)
        well_estimated_pruned.extend(we_prop_pruned)

        regret_history.extend(psp_sample_history)
        regret_pruned.extend(reg_prop_pruned)

        sample_history.extend(psp_sample_history)
        proportion_pruned.extend(prop_pruned)
        print(f'N: {n}')

    if ax is not None:
        if show_y_label:
            ax.set_ylabel(f'Proportion Pruned')
        # bennett_ecdf = ECDF(bennett_bound)
        # e_bennett_ecdf = ECDF(empirical_bennett_bound)
        # ax.fill_between(bennett_ecdf.x, bennett_ecdf.y, color='lightblue', step='post')
        # ax.fill_between([0, *sample_history], [0, *proportion_pruned], color='palegreen', step='post')
        # ax.fill_between(e_bennett_ecdf.x, e_bennett_ecdf.y, color='mistyrose', step='post')
        # sns.ecdfplot(bennett_bound, label='Bennett (Known Variance)', ax=ax, color='royalblue')
        sns.lineplot([0, *regret_history], [0, *regret_pruned], color='tomato', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP-REG', ci=None)
        # sns.scatterplot(regret_history, regret_pruned, color='tomato', ax=ax)
        sns.lineplot([0, *well_estimated_history], [0, *well_estimated_pruned], color='royalblue', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP-WE', ci=None)
        # sns.scatterplot(well_estimated_history, well_estimated_pruned, color='royalblue', ax=ax)
        # sns.ecdfplot(empirical_bennett_bound, label='PSP Data Complexity Upper Bound', ax=ax, color='tomato')
        # ax.set_xlim(0, max_sample_complexity)
        sns.lineplot([0, *sample_history], [0, *proportion_pruned], color='darkgreen', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP', ci=None)
        # sns.scatterplot(sample_history, proportion_pruned, color='darkgreen', ax=ax)
        if x_label_scale == 1:
            ax.set_xlabel('Number of Samples')
        else:
            ax.set_xlabel(f'Number of Samples ($\\cdot 10^{x_label_scale}$)')
        if title is not None:
            ax.set_title(title)

    # print(len(psp_sample_history))
    # print(len(proportion_pruned))
    #
    # ax.twiny()
    # sns.lineplot(psp_sample_history, proportion_pruned)
    # plt.xlim(start_x, start_x + max_var * multiplier)
    # plt.xlabel('Sample Complexity')
    # ax2.spines['top'].set_position(('axes', -0.15))
    # ax2.spines['top'].set_visible(False)
    # plt.tick_params(which='both', top=False)


def aaai_proportion_pruned_vs_samples_many_games(game_class, game_params, noise_params, delta=0.05, beta=1.5,
                                                 target_epsilon=5.0,
                                                 num_games=1, profiles=True, title=None, ax=None):
    # log_term = np.log(2) + np.log(game_size) + np.log(T) - np.log(delta)
    # multiplier = 2 * log_term / (target_epsilon ** 2)
    # start_x = multiplier * c * target_epsilon / 3

    # log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    # log_term_bennett = log_term - np.log(3) + np.log(2)

    # variance = empirical_game.variance_utils()
    # print(c)
    # bennett_bound = log_term_bennett * (c ** 2 / (variance * h(c * target_epsilon / variance)))
    # bennett_bound = np.amax(bennett_bound, axis=-1)  # Only care about profiles
    # bennett_bound = bennett_bound.reshape(-1)
    # empirical_bennett_bound = 1 + beta * ((8.0 / 3 + np.sqrt(
    #     4 + 2 / log_term)) * c * log_term / target_epsilon + 2 * variance * log_term / target_epsilon ** 2)
    # empirical_bennett_bound = np.amax(empirical_bennett_bound, axis=-1)  # Only care about profiles
    # empirical_bennett_bound = empirical_bennett_bound.reshape(-1)
    # var_ticks = 0.05
    # max_sample_complexity = np.amax(empirical_bennett_bound)
    # max_var = np.amax(variance)
    # print(max_var)
    # if max_var % var_ticks != 0:
    #     max_var += (var_ticks - max_var % var_ticks)

    sample_history = []
    proportion_pruned = []
    regret_history = []
    regret_pruned = []
    well_estimated_history = []
    well_estimated_pruned = []

    for n in range(num_games):
        empirical_game = construct_empirical_game(game_class(**game_params), **noise_params)

        c = empirical_game.get_c()
        T = schedule_length(c, target_epsilon, beta)
        if n == 0:
            print(f'NUM_ITERATIONS: {T}')
        game_size = np.prod(empirical_game.get_utils_shape())
        num_players = empirical_game.get_utils_shape()[-1]
        num_profiles = game_size / num_players

        psp_sample_history, psp_results = progressive_sampling_with_pruning(
            empirical_game,
            c=c,
            delta=delta,
            target_epsilon=target_epsilon,
            beta=beta,
            well_estimated_pruning=True,
            regret_pruning=True,
            old_regret_pruning=False,
            count_each_pruning_contribution=True,
            show_graphs_every=-1,
            verbose=2)

        if profiles:
            num_active_profiles = psp_results['num_active_profiles'][1:]
            reg_active_profiles = psp_results['regret_active_profiles'][1:]
            we_active_profiles = psp_results['well_estimated_active_profiles'][1:]
            prop_pruned = [1 - active_profiles / num_profiles for active_profiles in num_active_profiles]
            reg_prop_pruned = [1 - active_profiles / num_profiles for active_profiles in reg_active_profiles]
            we_prop_pruned = [1 - active_profiles / num_profiles for active_profiles in we_active_profiles]
        else:
            num_active_indices = psp_results['num_active_utils'][1:]
            reg_active_indices = psp_results['regret_active_utils'][1:]
            we_active_indices = psp_results['well_estimated_active_utils'][1:]
            prop_pruned = [1 - active_indices / game_size for active_indices in num_active_indices]
            reg_prop_pruned = [1 - active_indices / game_size for active_indices in reg_active_indices]
            we_prop_pruned = [1 - active_indices / game_size for active_indices in we_active_indices]

        print(f'Sample history: {psp_sample_history}')
        print(f'Prop Pruned: {prop_pruned}')

        well_estimated_history.extend(psp_sample_history)
        well_estimated_pruned.extend(we_prop_pruned)

        regret_history.extend(psp_sample_history)
        regret_pruned.extend(reg_prop_pruned)

        sample_history.extend(psp_sample_history)
        proportion_pruned.extend(prop_pruned)
        print(f'N: {n}')

    if ax is not None:
        ax.set_ylabel(f'Proportion Pruned ({"Profiles" if profiles else "Indices"})')
        # bennett_ecdf = ECDF(bennett_bound)
        # e_bennett_ecdf = ECDF(empirical_bennett_bound)
        # ax.fill_between(bennett_ecdf.x, bennett_ecdf.y, color='lightblue', step='post')
        # ax.fill_between([0, *sample_history], [0, *proportion_pruned], color='palegreen', step='post')
        # ax.fill_between(e_bennett_ecdf.x, e_bennett_ecdf.y, color='mistyrose', step='post')
        # sns.ecdfplot(bennett_bound, label='Bennett (Known Variance)', ax=ax, color='royalblue')
        sns.lineplot([0, *regret_history], [0, *regret_pruned], color='tomato', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP-REG')
        # sns.scatterplot(regret_history, regret_pruned, color='tomato', ax=ax)
        sns.lineplot([0, *well_estimated_history], [0, *well_estimated_pruned], color='royalblue', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP-WE')
        # sns.scatterplot(well_estimated_history, well_estimated_pruned, color='royalblue', ax=ax)
        # sns.ecdfplot(empirical_bennett_bound, label='PSP Data Complexity Upper Bound', ax=ax, color='tomato')
        # ax.set_xlim(0, max_sample_complexity)
        sns.lineplot([0, *sample_history], [0, *proportion_pruned], color='darkgreen', alpha=0.8,
                     drawstyle='steps-post', ax=ax, label='PSP')
        # sns.scatterplot(sample_history, proportion_pruned, color='darkgreen', ax=ax)
        ax.set_xlabel('Number of Samples')
        if title is not None:
            ax.set_title(title)

    # print(len(psp_sample_history))
    # print(len(proportion_pruned))
    #
    # ax.twiny()
    # sns.lineplot(psp_sample_history, proportion_pruned)
    # plt.xlim(start_x, start_x + max_var * multiplier)
    # plt.xlabel('Sample Complexity')
    # ax2.spines['top'].set_position(('axes', -0.15))
    # ax2.spines['top'].set_visible(False)
    # plt.tick_params(which='both', top=False)


def aaai_proportion_pruned_vs_samples_subplots(old_regret=False, num_iterations=5):
    n_rows = 1
    n_cols = 4
    figsize = (4 * 0.9 * n_cols, 4.65 * 0.9 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    target_epsilons = [3, 2, 1, 0.5]
    scales = [3, 3, 4, 4]
    num_goods = [2, 4, 6, 8]
    titles = [f'Target Epsilon: {e}' for e in target_epsilons]
    max_x_lim = 0
    bidding_game = BiddingGame(3, 4, 127, [0, 0.2, 0.4, 0.6, 0.8, 1], 1, random_distributions=False)
    for i in range(n_cols):
        row = i // n_cols
        col = i % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        elif n_cols > 1:
            ax = axes[col]
        else:
            ax = axes

        aaai_proportion_pruned_vs_samples(bidding_game,
                                          target_epsilon=target_epsilons[i], beta=1.1, profiles=True, title=titles[i],
                                          ax=ax, show_y_label=(row == 0 and col == 0),
                                          num_iterations=num_iterations, old_regret=old_regret, x_label_scale=scales[i])

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        elif n_cols > 1:
            ax = axes[col]
        else:
            ax = axes
        # ax.set_xlim(0, max_x_lim)
        ax.set_ylim(0, 1)
        ax.legend().set_visible(False)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('Proportion of Profiles Pruned Throughout Various Pruning Algorithms')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    fig.savefig('aaai_proportion_pruned_bidding_game.pdf', bbox_inches='tight')


def aaai_proportion_pruned_vs_samples_subplots_all_games():
    n_rows = 2
    n_cols = 4
    figsize = (4 * n_cols, 4.65 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    max_x_lim = 0
    for i in range(n_cols * n_rows):
        row = i // n_cols
        col = i % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        elif n_cols > 1:
            ax = axes[col]
        else:
            ax = axes

        aaai_proportion_pruned_vs_samples_many_games(
            game_gens[i][0], game_gens[i][1], {'complete_noise': 10,
                                               'noise_multiplier_distribution': scipy.stats.uniform,
                                               'noise_multiplier_args': {'loc': 0, 'scale': 1}},
            target_epsilon=0.2, beta=1.1, profiles=True, title=game_gens[i][2], ax=ax,
            num_games=100)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        elif n_cols > 1:
            ax = axes[col]
        else:
            ax = axes
        ax.set_xlim(0, max_x_lim)
        ax.set_ylim(0, 1)
        ax.legend().set_visible(False)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    fig.savefig('aaai_proportion_pruned_all_games.pdf', bbox_inches='tight')


def extreme_properties_graph(utils: np.ndarray, num_samples: int = 40, epsilon: float = 0.2, c: float = 2,
                             constrain_utils: bool = False):
    LOWER = 0.0000000001
    if constrain_utils:
        utils = np.maximum(LOWER, utils + c / 2)
    # lambdas = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 25.0]
    lambdas = [0, 0.25, 0.5, 0.75, 1, 1.25, 2.5, 5, 10, 25.0]
    ar_complete = None
    ag_complete = None
    ag_star_complete = None
    preliminary_results = extreme_properties(utils, rho=1, LAMBDA=1, constrain_utils=False)
    true_ag = preliminary_results['true_anarchy_gap']
    true_sg = preliminary_results['true_stability_gap']
    true_ar = preliminary_results['true_anarchy_ratio']
    true_sr = preliminary_results['true_stability_ratio']
    true_ag_lambda = []
    true_ag_star_lambda = []
    upper_ag_lambda = []
    lower_ag_lambda = []
    for LAMBDA in lambdas:
        print(LAMBDA)
        epsilons = scipy.stats.beta.rvs(size=(num_samples,) + utils.shape,
                                        **{'loc': -1, 'scale': 2, 'a': 2, 'b': 2}) * epsilon
        epsilons_2 = scipy.stats.beta.rvs(size=(num_samples,) + utils.shape,
                                          **{'loc': -1, 'scale': 2, 'a': 0.5, 'b': 0.5}) * epsilon
        ar = []
        sr = []
        ag = []
        ag_2 = []
        sg = []
        ag_star = []
        results = extreme_properties(utils, rho=1, LAMBDA=LAMBDA, constrain_utils=False)
        true_ag_lambda.append(results['anarchy_gap'])
        true_ag_star_lambda.append(results['anarchy_gap_star'])
        upper_ag_lambda.append(true_ag_lambda[-1] + 2 * (1 + LAMBDA) * epsilon)
        lower_ag_lambda.append(true_ag_lambda[-1] - 2 * (1 + LAMBDA) * epsilon)
        for i in range(num_samples):
            adjusted_utils = utils + epsilons[i]
            adjusted_utils_2 = utils + epsilons_2[i]
            if constrain_utils:
                adjusted_utils = np.maximum(LOWER, adjusted_utils)
                adjusted_utils_2 = np.maximum(LOWER, adjusted_utils_2)
            results = extreme_properties(adjusted_utils, rho=1, LAMBDA=LAMBDA, constrain_utils=False)
            results_2 = extreme_properties(adjusted_utils_2, rho=1, LAMBDA=LAMBDA, constrain_utils=False)
            ar.append(results['anarchy_ratio'])
            sr.append(results['stability_ratio'])
            ag.append(results['anarchy_gap'])
            ag_2.append(results_2['anarchy_gap'])
            sg.append(results['stability_gap'])
            ag_star.append(results['anarchy_gap_star'])
        ar_df = pd.DataFrame({'AR': ar})
        ag_df = pd.DataFrame({'AG': ag})
        ag_2_df = pd.DataFrame({'AG': ag_2})
        ag_df['Noise Model'] = 'parabolic'
        ag_2_df['Noise Model'] = 'arcsine'
        ag_star_df = pd.DataFrame({'AG*': ag_star})
        ar_df['lambda'] = LAMBDA
        ag_df['lambda'] = LAMBDA
        ag_2_df['lambda'] = LAMBDA
        ag_star_df['lambda'] = LAMBDA
        if ar_complete is None:
            ar_complete = ar_df
        else:
            ar_complete = pd.concat([ar_complete, ar_df])
        if ag_complete is None:
            ag_complete = pd.concat([ag_df, ag_2_df])
        else:
            ag_complete = pd.concat([ag_complete, ag_df, ag_2_df])
        if ag_star_complete is None:
            ag_star_complete = ag_star_df
        else:
            ag_star_complete = pd.concat([ag_star_complete, ag_star_df])

    # sns.violinplot(x='Sample Size', y='Regrets', hue='Bound Type', data=df_complete, scale='count', split=True, bw=0.2,
    #                inner='quartile')

    str_lambdas = list(map(str, lambdas))

    # plt.axhline(true_ar)
    # sns.violinplot(x='lambda', y='AR', data=ar_complete, scale='count', bw=0.2,
    #                inner='quartile')
    # plt.show()
    fig, ax = plt.subplots(1, 1)
    min_upper = min(upper_ag_lambda)
    loc = 0.09 + 0.9 * (upper_ag_lambda.index(min_upper) / len(upper_ag_lambda))
    plt.axhline(min_upper, xmin=loc, xmax=0.92, color='black', alpha=0.5, linestyle=':')
    plt.axhline(true_ag)
    plt.fill_between(str_lambdas, upper_ag_lambda, lower_ag_lambda
                     , color='lavender')
    sns.violinplot(x='lambda', y='AG', hue='Noise Model', data=ag_complete, split=True, scale='count', bw=0.2,
                   inner=None)
    sns.lineplot(x=str_lambdas, y=true_ag_lambda, color='black')
    sns.lineplot(x=str_lambdas, y=upper_ag_lambda, color='black', linestyle='--')
    sns.lineplot(x=str_lambdas, y=lower_ag_lambda, color='black', linestyle='--')
    plt.ylim(max(min(lower_ag_lambda), -2), upper_ag_lambda[0] * 1.05)
    plt.xlabel('Stability Parameter $\\Lambda$')
    plt.ylabel('AG$_\\Lambda$')
    plt.show()
    fig.savefig('anarchy_gap.pdf', bbox_inches='tight')
    # plt.axhline(true_ag)
    # sns.violinplot(x='lambda', y='AG*', data=ag_star_complete, scale='count', bw=0.2,
    #                inner='quartile')
    # # sns.lineplot(x=lambdas, y=true_ag_star_lambda, color='black')
    #
    # plt.show()


def bidding_game_variances():
    auction_type = 2
    num_player_types = 1
    num_good_types = 3
    fig, axes = plt.subplots(num_player_types, num_good_types, dpi=200,
                             figsize=(7 * num_good_types, 5 * num_player_types))
    players = [4, 3, 4]
    # goods = [1, 2, 3, 4, 5, 6, 8]
    goods = [1, 2, 3]
    max_max_variance = 0
    for players_idx in range(num_player_types):
        for goods_idx in range(num_good_types):
            num_players = players[players_idx]
            num_goods = goods[goods_idx]
            print(f'{num_players} Players, {num_goods} Goods')
            game = BiddingGame(num_players, num_goods, 127, [0, 0.2, 0.4, 0.6, 0.8, 1],
                               auction_type=auction_type,
                               random_distributions=True)

            alpha = 1
            beta = 1

            game.alphas = [alpha] * num_players
            game.betas = [beta] * num_players

            variances = game.variance_utils().reshape(-1)
            max_variance = np.amax(variances)
            max_max_variance = max(max_max_variance, max_variance)

            if num_player_types > 1 and num_good_types > 1:
                ax = axes[players_idx, goods_idx]
            elif num_player_types > 1:
                ax = axes[players_idx]
            elif num_good_types > 1:
                ax = axes[goods_idx]
            else:
                ax = axes

            ax.set_xlim(0, game.get_c() ** 2 / 4)
            ax.set_xlim(0, max_variance)

            sns.ecdfplot(variances, ax=ax, color='royalblue')
            ax.axvline(x=max_variance, ymin=0, ymax=1, color='orange')
            ax.set_title(
                f'{game.num_players} Players, {game.num_goods} Goods, $\\alpha={game.alphas}$, $\\beta={game.betas}$')

    for players_idx in range(num_player_types):
        for goods_idx in range(num_good_types):
            if num_player_types > 1 and num_good_types > 1:
                ax = axes[players_idx, goods_idx]
            elif num_player_types > 1:
                ax = axes[players_idx]
            elif num_good_types > 1:
                ax = axes[goods_idx]
            else:
                ax = axes
            ax.set_ylim(0, 1)
            ax.set_xlabel('Variance')
            ax.set_ylabel('Proportion')

    fig.suptitle(f'{"First" if auction_type == 1 else "Second"} Auction Bidding Game Variance CDFs')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    fig.savefig('bidding_game_variances.pdf', bbox_inches='tight')


# def proportion_remaining_vs_samples_aamas(empirical_game: EmpiricalGame, delta=0.05, beta=1.5, target_epsilon=5.0,
#                                           num_iterations=1, title=None, ax=None):
#     c = empirical_game.get_c()
#     T = np.ceil(np.log(3 * c / (4 * target_epsilon)) / np.log(beta)).astype('int')
#     game_size = np.prod(empirical_game.get_utils_shape())
#     num_players = empirical_game.get_utils_shape()[-1]
#     num_profiles = game_size / num_players
#
#     sample_history = []
#     proportion_pruned = []
#
#     for n in range(num_iterations):
#         psp_sample_history, psp_results = progressive_sampling_with_pruning(
#             empirical_game,
#             c=c,
#             delta=delta,
#             target_epsilon=target_epsilon,
#             beta=beta,
#             well_estimated_pruning=True,
#             regret_pruning=False,
#             show_graphs_every=-1)
#
#         num_active_profiles = psp_results['num_active_profiles'][1:]
#         proportion_pruned.extend(
#             [1 - active_profiles / num_profiles for active_profiles in num_active_profiles])
#         sample_history.extend(psp_sample_history)
#         print(f'N: {n}')
#
#     if ax is not None:
#         ax.set_ylabel('Proportion of Active Profiles')
#         bennett_ecdf = ECDF(bennett_bound)
#         e_bennett_ecdf = ECDF(empirical_bennett_bound)
#         bennett_ecdf.y = [1 - y for y in bennett_ecdf.y]
#         proportion_pruned = [1 - y for y in proportion_pruned]
#         e_bennett_ecdf.y = [1 - y for y in e_bennett_ecdf.y]
#         bennett_ecdf.x[0] = 0
#         e_bennett_ecdf.x[0] = 0
#         ax.fill_between(e_bennett_ecdf.x, e_bennett_ecdf.y, color='mistyrose', step='post')
#         ax.fill_between([0, *sample_history], [1, *proportion_pruned], color='palegreen', step='post')
#         ax.fill_between(bennett_ecdf.x, bennett_ecdf.y, color='lightblue', step='post')
#         sns.lineplot([0, *bennett_ecdf.x], [1, *bennett_ecdf.y], label='Bennett (Known Variance)', ax=ax,
#                      color='royalblue', drawstyle='steps-post')
#         sns.lineplot([0, *sample_history], [1, *proportion_pruned], color='darkgreen', alpha=0.8,
#                      drawstyle='steps-post', ax=ax)
#         sns.scatterplot(sample_history, [1, *proportion_pruned[:-1]], color='darkgreen', label='PSP', ax=ax)
#         sns.lineplot([0, *e_bennett_ecdf.x], [1, *e_bennett_ecdf.y], label='PSP Upper Bound', ax=ax,
#                      color='tomato', drawstyle='steps-post')
#         ax.set_xlim(0, max_sample_complexity)
#         ax.set_ylim(0, 1)
#         n = int(np.floor(np.log10(ax.get_xticks()[1])))
#         print(ax.get_xticks())
#         print(ax.get_xticklabels())
#         ax.set_xticklabels([f'{x / (10 ** n):.1f}' for x in ax.get_xticks()])
#         ax.set_xlabel(f'Number of Samples ($\\cdot 10^{n}$)')
#         print(ax.get_xticks())
#         print(ax.get_xticklabels())
#         if title is not None:
#             ax.set_title(title)
#     else:
#         plt.figure()
#         plt.ylabel('Proportion Pruned')
#
#         plt.subplot(111)
#         sns.ecdfplot(bennett_bound)
#         sns.scatterplot(sample_history, proportion_pruned)
#         sns.ecdfplot(empirical_bennett_bound)
#         plt.legend(
#             ['Bennett (Known Variance)', 'PSP', 'PSP Upper Bound'])
#         plt.xlim(0, max_sample_complexity)
#         plt.xlabel('Data Complexity')
#         if title is not None:
#             plt.title(title)
#         plt.show()
#
#     # print(len(psp_sample_history))
#     # print(len(proportion_pruned))
#     #
#     # ax.twiny()
#     # sns.lineplot(psp_sample_history, proportion_pruned)
#     # plt.xlim(start_x, start_x + max_var * multiplier)
#     # plt.xlabel('Sample Complexity')
#     # ax2.spines['top'].set_position(('axes', -0.15))
#     # ax2.spines['top'].set_visible(False)
#     # plt.tick_params(which='both', top=False)


def proportion_remaining_vs_samples_subplots():
    n_rows = 2
    n_cols = 2
    # figsize = (14.5, 4.15)
    figsize = (7.25, 7.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    alphas = [0.5, 1.5, 3, 5]
    betas = [3, 3, 1.5, 0.5]
    titles = [
        'Beta(0.5, 3); Mostly low variance',
        'Beta(1.5, 3); Mostly medium variance',
        'Beta(3, 1.5); Mostly high variance',
        'Beta(5, 0.5); Almost all high variance'
    ]
    max_x_lim = 0
    for i in range(4):
        row = i // n_cols
        col = i % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        proportion_remaining_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
                                                                 complete_noise=10,
                                                                 noise_distribution=scipy.stats.bernoulli,
                                                                 noise_args={'p': 0.5},
                                                                 noise_multiplier_distribution=scipy.stats.beta,
                                                                 noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                                        'a': alphas[i],
                                                                                        'b': betas[i]}),
                                        target_epsilon=0.2, delta=0.05, beta=1.1, title=titles[i], ax=ax)

        max_x_lim = max(max_x_lim, ax.get_xlim()[1])
        print(f'{i + 1} Done')
    for i in range(4):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_xlim(0, max_x_lim)
        ax.legend().set_visible(False)
        if col > 0:
            ax.set_ylabel('')
        if i == 3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.show()
    fig.savefig('proportion_remaining.pdf', bbox_inches='tight')


def eps_vs_sample_complexities_regret_pruning(empirical_game: EmpiricalGame, delta=0.05, beta=1.1,
                                              num_algo_iterations=1, algo_indices=None, num_epsilon=40,
                                              file_name=''):
    c = float(empirical_game.get_c())
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players
    true_utils = empirical_game.expected_utilities()
    variance = empirical_game.variance_utils()
    wimpy_variance = np.amax(variance)

    epsilons = [c / n for n in range(2, num_epsilon)]

    psp_complexities = []
    psp_inverse_epsilons = []

    pruning_criterias = [
        [well_estimated_pruning_criteria],
        [well_estimated_pruning_criteria, regret_pruning_old_criteria_plus],
        [well_estimated_pruning_criteria, regret_pruning_plus],
        [well_estimated_pruning_criteria, regret_pruning_plus],
        [well_estimated_pruning_criteria, regret_pruning_old_criteria_plus],
        [well_estimated_pruning_criteria, regret_pruning_mixed],
    ]

    sampling_schedules = [
        sampling_schedule_well_estimated,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
    ]

    wimpy_variance = [
        False,
        True,
        True,
        False,
        False,
        False
    ]

    pruning_names = [
        '$\\mathtt{PsWE}$',
        '$\\mathtt{PsReg}_0$',
        '$\\mathtt{PsReg}_{2\\epsilon}$',
        '$\\mathtt{PsReg}_{2\\epsilon}^{+}$',
        '$\\mathtt{PsReg}_{0}^{+}$',
        '$\\mathtt{PsRegM}$'
    ]

    pruning_upper_names = [
        '$\\mathtt{PsWE}$ Upper Bound',
        '$\\mathtt{PsReg}_{2\\epsilon}^{+}$ Upper Bound',
        '$\\mathtt{PsReg}_{0}^{+}$ Upper Bound',
        '$\\mathtt{PsRegM}$ Upper Bound'
    ]
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica"
    # })

    if algo_indices is None:
        algo_indices = [i for i in range(len(pruning_names))]

    for _ in algo_indices:
        psp_complexities.append([])
        psp_inverse_epsilons.append([])

    for n in range(num_algo_iterations):
        for type_index in algo_indices:
            sampling_schedule_func = sampling_schedules[type_index]
            pruning_criteria = pruning_criterias[type_index]
            wimpy = wimpy_variance[type_index]
            for eps in epsilons:
                print(f'Target Eps: {eps}')
                sampling_schedule = sampling_schedule_func(c, eps, delta, beta, game_size)
                psp_sample_history, psp_results = generalized_progressive_sampling(empirical_game, c, delta, eps,
                                                                                   sampling_schedule,
                                                                                   pruning_criteria,
                                                                                   wimpy_variance=wimpy,
                                                                                   verbose=0)
                num_active_profiles = psp_results['num_active_profiles']
                queries = psp_sample_history[0] * num_active_profiles[0]
                for i in range(1, len(psp_sample_history)):
                    queries += (psp_sample_history[i] - psp_sample_history[i - 1]) * num_active_profiles[i]
                psp_complexities[type_index].append(queries)
                # psp_inverse_epsilons.append(1 / psp_results['supremum_epsilon'][-1])
                psp_inverse_epsilons[type_index].append(1.0 / eps)
        print(f'Finished Algo Run {n}. Filename: {file_name}')

    # max_y = max(max(psp_complexities[i]) for i in range(len(psp_complexities)) if not wimpy_variance[i]) * 1.2

    inverse_epsilons = []
    we_complexities = []
    pure_0_query_complexities = []
    pure_query_complexities = []
    mixed_query_complexities = []

    for eps in epsilons:
        inverse_epsilons.append(1.0 / eps)
        we_complexities.append(ps_query_complexity(c, delta, eps, variance, beta, game_size))
        pure_query_complexities.append(
            ps_regret_pure_query_complexity(c, delta, eps, true_utils, 2 * eps, variance, beta, game_size)
        )
        pure_0_query_complexities.append(
            ps_regret_pure_query_complexity(c, delta, eps, true_utils, 0, variance, beta, game_size)
        )
        mixed_query_complexities.append(
            ps_regret_mixed_query_complexity(c, delta, eps, true_utils, variance, beta, game_size)
        )

    print(pure_query_complexities)
    print(pure_0_query_complexities)
    print(mixed_query_complexities)

    complexity_bounds = [we_complexities, pure_query_complexities, pure_0_query_complexities, mixed_query_complexities]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[1].set(xscale="log", yscale="log")

    for i in algo_indices:
        sns.lineplot(psp_inverse_epsilons[i], psp_complexities[i], ax=axes[0],
                     label=pruning_names[i])
        sns.lineplot(psp_inverse_epsilons[i], psp_complexities[i], ax=axes[1],
                     label=pruning_names[i])
    for i in range(len(pruning_upper_names)):
        sns.lineplot(inverse_epsilons, complexity_bounds[i], ax=axes[0],
                     label=pruning_upper_names[i])
    axes[0].set_xlabel('1/$\\varepsilon$')
    axes[0].grid()
    # axes.set_ylim(0, max_y)
    n = int(np.floor(np.log10(axes[0].get_yticks()[2])))
    axes[0].set_yticklabels([f'{y / (10 ** n):.1f}' for y in axes[0].get_yticks()])
    axes[0].set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    axes[0].set_title('Query Complexity vs $1/\\varepsilon$')
    axes[0].legend().set_visible(False)

    axes[1].set_xlabel('1/$\\varepsilon$')
    axes[1].grid()
    # axes.set_ylim(0, max_y)
    n = int(np.floor(np.log10(axes[1].get_yticks()[2])))
    axes[1].set_yticklabels([f'{y / (10 ** n):.1f}' for y in axes[1].get_yticks()])
    axes[1].set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    axes[1].set_title('Query Complexity vs $1/\\varepsilon$')
    axes[1].legend().set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    fig.savefig(f'regret_variations_{file_name}.pdf', bbox_inches='tight')

    with open(f'regret_variations_{file_name}.txt', 'w') as f:
        for i in algo_indices:
            f.write(pruning_names[i])
            f.write('\n')
            f.write(','.join(list(map(str, psp_inverse_epsilons[i]))))
            f.write('\n')
            f.write(','.join(list(map(str, psp_complexities[i]))))
            f.write('\n')
        for i in range(len(pruning_upper_names)):
            f.write(pruning_upper_names[i])
            f.write('\n')
            f.write(','.join(list(map(str, inverse_epsilons))))
            f.write('\n')
            f.write(','.join(list(map(str, complexity_bounds[i]))))
            f.write('\n')


def eps_vs_sample_complexities_regret_pruning_many_games(delta=0.05, beta=1.1, num_games=1, num_actions=20,
                                                         num_epsilon=80):
    psp_complexities = []
    psp_inverse_epsilons = []

    pruning_criterias = [
        [well_estimated_pruning_criteria],
        [well_estimated_pruning_criteria, regret_pruning_plus],
        [well_estimated_pruning_criteria, regret_pruning_old_criteria_plus],
        [well_estimated_pruning_criteria, regret_pruning_mixed],
    ]

    sampling_schedules = [
        sampling_schedule_well_estimated,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
        sampling_schedule_regret_geometric,
    ]

    wimpy_variance = [
        False,
        False,
        False,
        False
    ]

    pruning_names = [
        'PS-WE',
        'PS-REG+ $(\\gamma^* = 2\\epsilon)$',
        'PS-REG+ $(\\gamma^* = 0)$',
        'PS-REG-M'
    ]

    pruning_names = [
        '$\\mathtt{PsWE}$',
        '$\\mathtt{PsReg}_{2\\epsilon}^{+}$',
        '$\\mathtt{PsReg}_{0}^{+}$',
        '$\\mathtt{PsRegM}$'
    ]

    for _ in range(len(pruning_names)):
        psp_complexities.append([])
        psp_inverse_epsilons.append([])

    for n in range(num_games):
        empirical_game = construct_empirical_game(congestion_game(3, 4, 4),
                                                  complete_noise=10,
                                                  noise_distribution=scipy.stats.bernoulli,
                                                  noise_args={'p': 0.5},
                                                  noise_multiplier_distribution=scipy.stats.beta,
                                                  noise_multiplier_args={'loc': 0, 'scale': 1,
                                                                         'a': 1.5,
                                                                         'b': 3})

        c = float(empirical_game.get_c())
        game_size = np.prod(empirical_game.get_utils_shape())
        epsilons = [c / n for n in range(2, num_epsilon, 2)]

        for type_index in range(len(pruning_criterias)):
            sampling_schedule_func = sampling_schedules[type_index]
            pruning_criteria = pruning_criterias[type_index]
            wimpy = wimpy_variance[type_index]
            for eps in epsilons:
                print(f'Target Eps: {eps}')
                sampling_schedule = sampling_schedule_func(c, eps, delta, beta, game_size)
                psp_sample_history, psp_results = generalized_progressive_sampling(empirical_game, c, delta, eps,
                                                                                   sampling_schedule,
                                                                                   pruning_criteria,
                                                                                   wimpy_variance=wimpy,
                                                                                   verbose=0)
                num_active_profiles = psp_results['num_active_profiles']
                queries = psp_sample_history[0] * num_active_profiles[0]
                for i in range(1, len(psp_sample_history)):
                    queries += (psp_sample_history[i] - psp_sample_history[i - 1]) * num_active_profiles[i]
                psp_complexities[type_index].append(queries)
                # psp_inverse_epsilons.append(1 / psp_results['supremum_epsilon'][-1])
                psp_inverse_epsilons[type_index].append(1 / eps)
        print('Finished Algo Run')

    for i in range(len(psp_complexities[0])):
        for j in range(1, len(pruning_names)):
            psp_complexities[j][i] /= psp_complexities[0][i]

    for i in range(len(psp_complexities[0])):
        psp_complexities[0][i] = 1.0

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    for i in range(len(pruning_names)):
        sns.lineplot(psp_inverse_epsilons[i], psp_complexities[i], ax=axes,
                     label=pruning_names[i])
    axes.set_xlabel('1/$\\varepsilon$')
    axes.set_ylim(0, 1.05)
    axes.set_ylabel(f'Query Complexity Ratio')
    axes.set_title('Query Complexity Relative to PS-WE')

    plt.tight_layout()
    plt.show()
    fig.savefig('regret_variations_2.pdf', bbox_inches='tight')

    with open('regret_variations_many_games_congestion_3_4_4.txt', 'w') as f:
        for i in range(len(pruning_names)):
            f.write(pruning_names[i])
            f.write('\n')
            f.write(','.join(list(map(str, psp_inverse_epsilons[i]))))
            f.write('\n')
            f.write(','.join(list(map(str, psp_complexities[i]))))
            f.write('\n')


def plot_eps_vs_sample_complexity_regret(file_names, game_names):
    N = len(file_names)
    names = []
    inverse_epsilons = []
    complexities = []
    for i in range(N):
        with open(file_names[i]) as file:
            lines = file.read().splitlines()
            names.append([])
            inverse_epsilons.append([])
            complexities.append([])
            for j in range(0, len(lines), 3):
                names[i].append(lines[j])
                inverse_epsilons[i].append(list(map(float, lines[j + 1].split(','))))
                complexities[i].append(list(map(float, lines[j + 2].split(','))))
    nr = 1
    fig, axes = plt.subplots(nr, N, figsize=(4 * 1.1 * N, 4.5 * 1.1 * nr))
    for i in range(N):
        if nr == 2:
            ax = axes[0, i]
        else:
            ax = axes[i]
        max_y = 1.04 * max(complexities[i][0])

        for j in range(len(names[i])):
            sns.lineplot(inverse_epsilons[i][j], complexities[i][j], ax=ax,
                         label=names[i][j])
        ax.set_xlabel('1/$\\varepsilon$')
        ax.grid()
        ax.set_ylim(0, max_y)
        n = int(np.floor(np.log10(ax.get_yticks()[2])))
        ax.set_yticklabels([f'{y / (10 ** n):.1f}' for y in ax.get_yticks()])
        ax.set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
        ax.set_title(f'{game_names[i]}')
        ax.legend().set_visible(False)

    if nr == 2:
        ref_ax = axes[0, 0]
    else:
        ref_ax = axes[0]
    handles, labels = ref_ax.get_legend_handles_labels()
    handles = [handles[i] for i in [1, 4, 2, 3, 0, 5]]
    labels = [labels[i] for i in [1, 4, 2, 3, 0, 5]]
    fig.legend(handles, labels, loc='lower center', ncol=3)
    # fig.suptitle('Query Complexity vs $1/\\varepsilon$')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()
    fig.savefig(f'regret_variations_final.pdf', bbox_inches='tight')


def plot_eps_vs_sample_complexity_regret_many_games(file_names, game_names, divide, colors):
    N = len(file_names)
    names = []
    inverse_epsilons = []
    complexities = []
    for i in range(N):
        with open(file_names[i]) as file:
            lines = file.read().splitlines()
            names.append([])
            inverse_epsilons.append([])
            complexities.append([])
            for j in range(0, len(lines), 3):
                if lines[j] == '$\\mathtt{PsReg}_0$' or lines[j] == '$\\mathtt{PsReg}_{2\\epsilon}$':
                    continue
                names[i].append(lines[j])
                inverse_epsilons[i].append(list(map(float, lines[j + 1].split(','))))
                complexities[i].append(list(map(float, lines[j + 2].split(','))))
    for i in range(N):
        if divide[i]:
            for j in range(len(complexities[i]))[::-1]:
                complexities[i][j] = [complexities[i][j][k] / complexities[i][0][k] for k in
                                      range(len(complexities[i][0]))]
    nr = 1
    fig, axes = plt.subplots(nr, N, figsize=(4 * 1.1 * N, 4.5 * 1.1 * nr))
    for i in range(N):
        if nr == 2:
            ax = axes[0, i]
        else:
            ax = axes[i]
        max_y = 1.04 * max(complexities[i][0])

        for j in range(len(names[i])):
            error_bar = None
            if i == 0 and j > 0:
                error_bar = lambda x: (x.min() - 0.01, x.max() + 0.01) if x.min() < 0.75 else\
                    (x.min() - 0.005, x.max() + 0.005) if x.min() < 0.8 else (x.min(), x.max())
            elif i == 1:
                error_bar = lambda x: (x.min(), x.max())
            sns.lineplot(x=inverse_epsilons[i][j], y=complexities[i][j], ax=ax,
                         label=names[i][j], errorbar=error_bar, color=colors[j])
        ax.set_xlabel('1/$\\varepsilon$')
        ax.grid()
        ax.set_ylim(0, max_y)
        # n = int(np.floor(np.log10(ax.get_yticks()[2])))
        # ax.set_yticklabels([f'{y / (10 ** n):.1f}' for y in ax.get_yticks()])
        ax.set_ylabel(f'Query Complexity Ratio (w.r.t. PsWE)')
        ax.set_title(f'{game_names[i]}')
        ax.legend().set_visible(False)

    if nr == 2:
        ref_ax = axes[0, 0]
    else:
        ref_ax = axes[0]
    handles, labels = ref_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    # fig.suptitle('Query Complexity vs $1/\\varepsilon$')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
    fig.savefig(f'regret_variations_many_games_final.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_eps_vs_sample_complexity_regret(['./regret_variations_random_zero_sum.txt',
                                          './regret_variations_congestion_3_4_4.txt'],
                                         ['2-Player Zero-Sum Game (40 Actions)',
                                          '3-Player Congestion Game (4 Facilities)'])
    # plot_eps_vs_sample_complexity_regret_many_games(['./regret_variations_random_zero_sum.txt',
    #                                                  './regret_variations_many_games_congestion_3_4_4.txt'],
    #                                                 ['2-Player Zero-Sum Game (40 Actions)',
    #                                                  '3-Player Congestion Game (4 Facilities)'],
    #                                                 divide = [True, False],
    #                                                 colors=['royalblue', 'red', 'mediumpurple', 'brown'])
    # game = random_zero_sum(80, c=2)
    # print(game.utils.shape)
    # precision_recall_by_batch(lambda: construct_empirical_game(congestion_game(3, 2, 2), complete_noise=10,
    #                                                            noise_multiplier_distribution=scipy.stats.uniform,
    #                                                            noise_multiplier_args={'loc': 0, 'scale': 1}),
    #                           batch_size=2000,
    #                           num_batches=100,
    #                           num_games=100)
    # eps_vs_sample_complexities_regret_pruning_many_games(num_games=10, num_actions=40, num_epsilon=100)
    # eps_vs_sample_complexities_regret_pruning(
    #     construct_empirical_game(random_zero_sum(40, 4),
    #                              complete_noise=10,
    #                              noise_distribution=scipy.stats.bernoulli,
    #                              noise_args={'p': 0.5},
    #                              noise_multiplier_distribution=scipy.stats.beta,
    #                              noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                     'a': 1.5,
    #                                                     'b': 3}),
    #     num_algo_iterations=10, num_epsilon=100
    # )
    # eps_vs_sample_complexities_regret_pruning(
    #     construct_empirical_game(random_zero_sum(40, 4),
    #                              complete_noise=10,
    #                              noise_distribution=scipy.stats.bernoulli,
    #                              noise_args={'p': 0.5},
    #                              noise_multiplier_distribution=scipy.stats.beta,
    #                              noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                     'a': 1.5,
    #                                                     'b': 3}),
    #     num_algo_iterations=1, num_epsilon=20, file_name='zero_sum_40'
    # )
    # with open('./regret_variations_third_run.txt') as file:
    #     for line in f7ile:
    #         line = line.split(',')
    #         psp_complexities = [[], [], [], [], [], []]
    #         psp_inverse_epsilons = [[], [], [], [], [], []]
    #         pruning_names = [
    #             'PS-WE',
    #             'PS-REG-0',
    #             'PS-REG $(\\gamma^* = 2\\epsilon)$',
    #             'PS-REG+ $(\\gamma^* = 2\\epsilon)$',
    #             'PS-REG+ $(\\gamma^* = 0)$',
    #             'PS-REG-M'
    #         ]
    #         wimpy_variance = [
    #             False,
    #             True,
    #             True,
    #             False,
    #             False,
    #             False
    #         ]
    #
    #         algo_indices = [i for i in range(6)]
    #
    #         for type_idx in range(6):
    #             psp_inverse_epsilons[type_idx] = list(map(float, line[98 * 20 * type_idx: 98 * 20 * type_idx + 98 * 10]))
    #             psp_complexities[type_idx] = list(map(float, line[98 * 20 * type_idx + 98 * 10: 98 * 20 * (type_idx + 1)]))
    #
    #         max_y = max(max(psp_complexities[i]) for i in range(len(psp_complexities)) if not wimpy_variance[i]) * 1.2
    #
    #         fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    #
    #         for i in algo_indices:
    #             sns.lineplot(psp_inverse_epsilons[i], psp_complexities[i], ax=axes,
    #                          label=pruning_names[i])
    #         axes.set_xlabel('1/$\\varepsilon$')
    #         axes.set_ylim(0, max_y)
    #         n = int(np.floor(np.log10(axes.get_yticks()[2])))
    #         axes.set_yticklabels([f'{y / (10 ** n):.1f}' for y in axes.get_yticks()])
    #         axes.set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    #         axes.set_title('Query Complexity vs $1/\\varepsilon$')
    #
    #         plt.tight_layout()
    #         plt.show()
    #         fig.savefig('regret_variations_3.pdf', bbox_inches='tight')

    # eps_vs_sample_complexities_regret_pruning(
    #     construct_empirical_game(congestion_game(3, 3, 2),
    #                              complete_noise=20,
    #                              noise_distribution=scipy.stats.bernoulli,
    #                              noise_args={'p': 0.5},
    #                              noise_multiplier_distribution=scipy.stats.beta,
    #                              noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                     'a': 1.5,
    #                                                     'b': 3}),
    #     num_algo_iterations=5, algo_indices=[0, 1], num_epsilon=60
    # )
    # aaai_proportion_pruned_vs_samples_subplots(old_regret=True, num_iterations=5)
    # eps_vs_query_complexities_amy_bidding_game(fig_scale=4)
    # bidding_game_variances()
    # proportion_remaining_vs_samples_subplots_auctions()
    # fig, ax = plt.subplots(1,1)
    # aaai_proportion_pruned_vs_samples_many_games(
    #     game_gens[6][0], game_gens[6][1], {'complete_noise': 10,
    #                                        'noise_multiplier_distribution': scipy.stats.uniform,
    #                                        'noise_multiplier_args': {'loc': 0, 'scale': 1}},
    #     target_epsilon=0.2, beta=1.1, profiles=True, title=game_gens[6][2], ax=ax,
    #     num_games=100)
    # aaai_proportion_pruned_vs_samples_subplots_all_games()
    # empirical_game = BiddingGame(4, 2, 127, [0.2, 0.4, 0.6, 0.8], 1)
    # true_regrets = np.amax(regrets_matrix(empirical_game.expected_utilities()), axis=-1).reshape(-1)
    # regret_sorted_indices = np.flip(np.argsort(true_regrets))
    # true_nash = (true_regrets == 0).astype(int)
    # print(true_nash)
    # gs_frequency_of_eps_pure_equilibria(
    #     lambda: BiddingGame(3, 4, 127, [0, 0.33, 0.66, 1], 1, random_distributions=True),
    #     num_games=200,
    #     sample_sizes=[100, 200, 1000, 4000],
    #     batch_size=100, smart_candidate_eqa=True, delta=0.05,)
    # gs_frequency_of_eps_pure_equilibria(
    #     lambda: construct_empirical_game(congestion_game(3, 2, 2), complete_noise=10,
    #                                      noise_multiplier_distribution=scipy.stats.uniform,
    #                                      noise_multiplier_args={'loc': 0, 'scale': 1}),
    #     num_games=10,
    #     sample_sizes=[10000, 30000, 100000, 300000],
    #     batch_size=10000)
    # c = 4
    # game = congestion_game(3, 3, c)
    # utils = game.utils
    # print(iterated_dominance(utils))
    # extreme_properties_graph(utils, num_samples=600, c=c, constrain_utils=True)
    # aaai_proportion_pruned_vs_samples_subplots()
    # proportion_remaining_vs_samples_subplots()
    # power_mean_welfare_error_sigmoidal(congestion_game(3, 3, 2))
    # eps_vs_sample_complexities_amy()
    # eps_vs_sample_complexities(construct_empirical_game(congestion_game(3, 3, c=2),
    #                                                     complete_noise=10,
    #                                                     noise_distribution=scipy.stats.bernoulli,
    #                                                     noise_args={'p': 0.5},
    #                                                     noise_multiplier_distribution=scipy.stats.beta,
    #                                                     noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                                            'a': 1.5,
    #                                                                            'b': 3}),
    #                            batch_size=200, num_algo_iterations=10, show_error_bars=True)
    # eps_vs_sample_complexities_final_subplots_auctions()
    # proportion_pruned_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
    #                                                       complete_noise=10,
    #                                                       noise_distribution=scipy.stats.bernoulli,
    #                                                       noise_args={'p': 0.5},
    #                                                       noise_multiplier_distribution=scipy.stats.beta,
    #                                                       noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                                              'a': 0.5,
    #                                                                              'b': 1.5}),
    #                              target_epsilon=0.2, beta=1.1)
    # proportion_regret_pruned_vs_samples(construct_empirical_game(random_zero_sum(80, c=2),
    #                                                              complete_noise=10,
    #                                                              noise_distribution=scipy.stats.bernoulli,
    #                                                              noise_args={'p': 0.5},
    #                                                              noise_multiplier_distribution=scipy.stats.beta,
    #                                                              noise_multiplier_args={'loc': 0, 'scale': 1,
    #                                                                                     'a': 0.5,
    #                                                                                     'b': 1.5}),
    #                                     target_epsilon=0.2, beta=1.1)
    # proportion_pruned_vs_samples_subplots()
    # power_mean_welfare_error_sigmoidal(congestion_game(3, 3, c=2))
    # proportion_pruned_vs_samples_subplots()
    # regrets_distributions_psp(construct_empirical_game(congestion_game(3, 3, c=2),
    #                                                    complete_noise=1,
    #                                                    noise_multiplier_distribution=scipy.stats.beta,
    #                                                    noise_multiplier_args={'loc': 0, 'scale': 1, 'a': 1, 'b': 1}
    #                                                    ),
    #                           target_epsilon=0.05)
    # average_regret_distribution_many_games(num_game_samples=100, remove_zero_regret=True)
    # average_regret_distribution_many_games(num_game_samples=100)
    # # average_true_regret_distribution(game_gens[0][0], game_gens[0][1], 20, num_individual_cdfs=5,
    #                                  title='Congestion Game (324 profiles)')
    # true_regret_distribution(congestion_game(3, 3, c=2), 'Congestion Game (343 profiles)')
    # true_regret_distribution(grab_the_dollar(23, c=2), 'GrabTheDollar (529 profiles)')
    # ps_vs_psp_sample_complexities(construct_empirical_game(congestion_game(3, 4, c=2),
    #                                                        complete_noise=1,
    #                                                        noise_multiplier_distribution=scipy.stats.uniform),
    #                               target_epsilons=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25])
    # gs_vs_psp_sample_complexities(construct_empirical_game(congestion_game(3, 4, c=200),
    #                                                        complete_noise=100,
    #                                                        noise_multiplier_distribution=scipy.stats.uniform),
    #                               target_epsilons=[5, 5.5, 6, 7, 8.5, 10, 15, 20, 30, 40],
    #                               batch_sizes=[100, 100, 100, 100, 100, 50, 10, 5, 5, 5])
    # gs_vs_psp_single_run(construct_empirical_game(congestion_game_restricted(5, 3, 2, c=10),
    #                                               complete_noise=2,
    #                                               noise_multiplier_distribution=None), 1,
    #                      gs_batch_size=5, gs_initial_batch_size=5)
    # frequency_of_eps_pure_equilibria(construct_empirical_game(congestion_game_restricted(5, 3, 2, c=200),
    #                                                           complete_noise=
    # frequency_of_eps_pure_equilibria_by_batch(construct_empirical_game(congestion_game_restricted(5, 3, 2, c=10),
    #                                                                    complete_noise=2,
    #                                                                    noise_multiplier_distribution=scipy.stats.uniform),
    #                                           batch_size=100)
    # frequency_of_eps_pure_equilibria_by_batch(construct_empirical_game(congestion_game(2, 3, c=10),
    #                                                                    complete_noise=2,
    #                                                                    noise_multiplier_distribution=None),
    #                                           batch_size=100)

    # rows = 1
    # cols = 4
    # fig, axs = plt.subplots(rows, cols, dpi=200, figsize=(16, 4.65))
    #
    # true_nash_frequency = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0,
    #      0, 0., 0., 0., 0., 0, 0., 36., 100.]
    # eps_nash_frequencies = [
    #     [0., 2., 4., 5., 6., 6., 6., 8., 7., 11., 12., 12., 12., 15.,
    #      15., 17., 21., 57., 63., 68., 67., 73., 81., 99., 100., 64., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 7.,
    #      8., 11., 12., 16., 18., 62., 64., 64., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0,
    #     0, 0., 0., 0., 0., 9., 10., 49., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0,
    #      0, 0., 0., 0., 0., 1., 1., 37., 0.]
    # ]
    # sample_sizes = [100, 200, 500, 1000]
    # num_games = 100
    #
    # for idx in range(4):
    #     eps_nash_frequency = eps_nash_frequencies[idx]
    #
    #     row = idx // cols
    #     col = idx % cols
    #
    #     if rows > 1 and cols > 1:
    #         ax = axs[row, col]
    #     elif cols > 1:
    #         ax = axs[col]
    #     elif rows > 1:
    #         ax = axs[rows]
    #     else:
    #         ax = axs
    #
    #     # ax.fill_between(np.arange(0, len(eps_nash_frequency) + 1, 0.01), np.amax(eps_nash_frequency[:-1]), num_games,
    #     #                 color='lightgreen')
    #     # ax.axhline(np.amax(eps_nash_frequency[:-1]), linestyle='--', color='seagreen')
    #     ax.bar(list(range(1, len(eps_nash_frequency) + 1)), eps_nash_frequency, width=0.65, color='royalblue',
    #            label='Spurious Equilibria',
    #            bottom=true_nash_frequency)
    #     ax.bar(list(range(1, len(true_nash_frequency) + 1)), true_nash_frequency, width=0.65, color='orangered',
    #            label='True Nash Equilibria')
    #     ax.set_xticks(list(range(1, len(eps_nash_frequency) + 1)))
    #     ax.set_xticklabels([])
    #     ax.set_title(f'{sample_sizes[idx]} Samples')
    #     ax.set_xlabel('Strategy Profiles')
    #     if row == 0 and col == 0:
    #         ax.set_ylabel('Num Games')
    #     ax.set_xlim(0, len(eps_nash_frequency) + 1)
    #     ax.set_ylim(0, num_games)
    #
    #     ax.legend().set_visible(False)
    #     if idx == 0:
    #         handles, labels = ax.get_legend_handles_labels()
    #         fig.legend(handles, labels, loc='lower center', ncol=2)
    #
    # fig.suptitle('Frequency of Candidate Nash Equilibria')
    # plt.tight_layout(rect=[0, 0.08, 1, 1])
    # plt.show()
