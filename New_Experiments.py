import Algorithms
from Algorithms import *
from BiddingGame import BiddingGame
from Games import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from EmpiricalGame import EmpiricalGame


def eps_vs_sample_complexities_regret_pruning(empirical_game_gen, delta=0.05, beta=1.1,
                                              num_games=1, algo_indices=None, num_epsilon=40,
                                              file_name=''):
    empirical_game = empirical_game_gen()

    c = float(empirical_game.get_c())
    game_size = np.prod(empirical_game.get_utils_shape())
    num_players = empirical_game.get_utils_shape()[-1]
    num_profiles = game_size / num_players
    # true_utils = empirical_game.expected_utilities()
    # variance = empirical_game.variance_utils()
    # wimpy_variance = np.amax(variance)

    epsilons = [c / 2, c / 3, c / 4]
    epsilons.extend([c / n for n in range(5, num_epsilon + 1, 5)])

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

    if algo_indices is None:
        algo_indices = [i for i in range(len(pruning_names))]

    for _ in algo_indices:
        psp_complexities.append([])
        psp_inverse_epsilons.append([])

    for n in range(num_games):
        if n > 0:
            empirical_game = empirical_game_gen()

        for type_index in algo_indices:
            sampling_schedule_func = sampling_schedules[type_index]
            pruning_criteria = pruning_criterias[type_index]
            wimpy = wimpy_variance[type_index]
            for eps_idx in range(len(epsilons)):
                eps = epsilons[eps_idx]
                print(f'Target Eps ({eps_idx}): {eps}')
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

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax.set(xscale="log", yscale="log")

    for i in algo_indices:
        sns.lineplot(x=psp_inverse_epsilons[i], y=psp_complexities[i], ax=ax,
                     label=pruning_names[i])
    ax.set_xlabel('1/$\\varepsilon$')
    ax.grid()
    # axes.set_ylim(0, max_y)
    n = int(np.floor(np.log10(ax.get_yticks()[2])))
    ax.set_yticklabels([f'{y / (10 ** n):.1f}' for y in ax.get_yticks()])
    ax.set_ylabel(f'Query Complexity ($\\cdot 10^{n}$)')
    ax.set_title('Query Complexity vs $1/\\varepsilon$')
    ax.legend().set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=(len(algo_indices) + 1) // 2)

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


def plot_eps_vs_sample_complexity_regret(file_names, game_names, c=1):
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
                inverse_epsilons[i].append(list(map(lambda x: float(x) * c, lines[j + 1].split(','))))
                complexities[i].append(list(map(float, lines[j + 2].split(','))))

    nr = 1
    fig, axes = plt.subplots(nr, N, figsize=(4 * 1.1 * N, 4.5 * 1.1 * nr))
    for i in range(N):
        if nr == 2:
            ax = axes[0, i]
        else:
            ax = axes[i]
        max_y = 1.04 * max(complexities[i][0])
        max_x = 1.04 * max(inverse_epsilons[i][0])

        for j in range(len(names[i])):
            sns.lineplot(x=inverse_epsilons[i][j], y=complexities[i][j], ax=ax,
                         label=names[i][j], errorbar=lambda x: (x.min(), x.max()))
        ax.set_xlabel('c/$\\varepsilon$')
        ax.grid()
        ax.set_xlim(0, max_x)
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
    fig.legend(handles, labels, loc='lower center', ncol=6)
    # fig.suptitle('Query Complexity vs $1/\\varepsilon$')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    fig.savefig(f'regret_variations_bidding_game.pdf', bbox_inches='tight')


plot_eps_vs_sample_complexity_regret([
    'regret_variations_2_players_6_goods_3_actions_10_runs.txt',
    'regret_variations_2_players_6_goods_5_actions_pos_10_runs.txt'],
    [
        '2 players; 3 strategies',
        '2 players; 5 (non-zero) strategies'],
    c=127 * 6)

# shading_factors = list(np.arange(0.2, 1, step=0.2))
# shading_factors.append(1.0)
# print(shading_factors)
#
# eps_vs_sample_complexities_regret_pruning(
#     lambda: BiddingGame(2, 6, 127, shading_factors,
#                         1, False),
#     num_games=10, num_epsilon=250, file_name='2_players_6_goods_5_actions_pos_10_runs'
# )

# game = BiddingGame(2, 6, 127, shading_factors,
#                    1, False)
# c = game.get_c()
# delta = 0.05
# eps = 10
# beta = 1.1
# game_size = np.prod(game.get_utils_shape())
# sampling_schedule = sampling_schedule_regret_geometric(c, eps, delta, beta, game_size)
#
# samp, res = generalized_progressive_sampling(game, c, delta, eps, sampling_schedule,
#                                              [well_estimated_pruning_criteria, regret_pruning_mixed],
#                                              wimpy_variance=False, verbose=0)
# num_active_profiles = res['num_active_profiles']
# queries = samp[0] * num_active_profiles[0]
# for i in range(1, len(samp)):
#     queries += (samp[i] - samp[i - 1]) * num_active_profiles[i]
# print(queries)

# # var = game.variance_utils(10000).reshape(-1)
# reg = Algorithms.regrets_matrix(game.expected_utilities(10000)).reshape(-1)
# # sns.ecdfplot(var)
# plt.show()
#
# sns.ecdfplot(reg)
# plt.show()
