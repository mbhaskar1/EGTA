import itertools
import math
import numpy as np
import typing
from typing import List, Union, Callable
from nptyping import NDArray, Bool
from matplotlib import pyplot as plt

import EGTA
from EGTA import h
from GamutGame import GamutGame
from NoisyGame import NoisyGame, Noise, NoiseFactor, NoiseType
from EmpiricalGame import EmpiricalGame
import scipy.stats


class Strategy:
    def __init__(self, probabilities: List[float]):
        self.probabilities = probabilities
        assert sum(probabilities) == 1, f'Probabilities for strategy {probabilities} sums to {sum(probabilities)} != 1'


def power_mean_welfare_matrix(utils: np.ndarray, rho: float, shift: float = 0, constrain_min: bool = True):
    if constrain_min:
        utils = np.maximum(0, utils + shift)
    else:
        utils = utils + shift
    if rho == 0:
        return np.power(np.prod(utils, axis=-1), 1.0 / (len(utils.shape) - 1))
    if rho == np.inf:
        return np.amax(utils, axis=-1)
    if rho == -np.inf:
        return np.amin(utils, axis=-1)
    return np.power(np.mean(np.power(utils, rho), axis=-1), 1 / rho)


def extreme_properties(utils: np.ndarray, rho: float, LAMBDA: float, shift: float = 0, constrain_utils: bool = True):
    welfare = power_mean_welfare_matrix(utils, rho, shift)
    max_welfare = np.amax(welfare)

    regrets = np.amax(regrets_matrix(utils), axis=-1)
    min_regret = np.amin(regrets)
    excess_regrets = regrets - min_regret

    md_lambda = np.amin(welfare + LAMBDA * regrets)
    md_lambda_star = np.amin(welfare + LAMBDA * excess_regrets)
    mc_lambda = np.amax(welfare - LAMBDA * regrets)

    equilibria = (regrets == 0)
    md = np.amin(welfare, where=equilibria, initial=1000)
    mc = np.amax(welfare, where=equilibria, initial=-1000)

    return {
        'anarchy_ratio': max_welfare / md_lambda,
        'stability_ratio': max_welfare / mc_lambda,
        'anarchy_gap': max_welfare - md_lambda,
        'anarchy_gap_star': max_welfare - md_lambda_star,
        'stability_gap': max_welfare - mc_lambda,
        'true_anarchy_gap': max_welfare - md,
        'true_stability_gap': max_welfare - mc,
        'true_anarchy_ratio': max_welfare / md,
        'true_stability_ratio': max_welfare / mc
    }


def regret(utils: np.ndarray, opponents_profile: List, player: int, player_actions: List[int] = None):
    opponents_profile.append(player)
    opponents_profile[player] = slice(utils.shape[player])
    best_response = np.amax(utils[tuple(opponents_profile)])
    if player_actions is not None:
        opponents_profile[player] = player_actions
    return best_response - utils[tuple(opponents_profile)]


def regrets_matrix(utils: np.ndarray):
    regrets = np.empty(utils.shape)
    for player in range(len(utils.shape) - 1):
        best_response_matrix = np.expand_dims(np.amax(utils[..., player], axis=player), axis=player)
        regrets[..., player] = best_response_matrix - utils[..., player]
    return regrets


def eps_nash_equilibria(utils: np.ndarray, epsilon: float):
    """Returns numpy array with elements being 1 if they correspond to epsilon nash equilibria and 0 otherwise

    :param utils: Utilities numpy array
    :param epsilon: Epsilon
    :return: Boolean numpy array
    """
    regrets = regrets_matrix(utils)
    return (np.amax(np.maximum(0, regrets - epsilon), axis=-1) == 0).astype(int)


def regrets_upper_bound_matrix(utils: np.ndarray, epsilon_matrix: np.ndarray):
    regrets = np.empty(utils.shape)
    for player in range(len(utils.shape) - 1):
        best_response_matrix = np.expand_dims(np.amax(utils[..., player] + epsilon_matrix[..., player], axis=player),
                                              axis=player)
        regrets[..., player] = best_response_matrix - (utils[..., player] - epsilon_matrix[..., player])
    return np.maximum(0, regrets)


def regrets_lower_bound_matrix(utils: np.ndarray, epsilon_matrix: np.ndarray = None, epsilon: float = None):
    assert (epsilon_matrix is None) != (epsilon is None)
    regrets = np.empty(utils.shape)
    for player in range(len(utils.shape) - 1):
        if epsilon_matrix is not None:
            best_response_matrix = np.expand_dims(
                np.amax(utils[..., player] - epsilon_matrix[..., player], axis=player),
                axis=player)
            regrets[..., player] = best_response_matrix - (utils[..., player] + epsilon_matrix[..., player])
        else:
            best_response_matrix = np.expand_dims(np.amax(utils[..., player] - epsilon, axis=player), axis=player)
            regrets[..., player] = best_response_matrix - (utils[..., player] + epsilon)
    return np.maximum(0, regrets)


def lower_bound_epsilon_nash_equilibria(utils: np.ndarray, epsilon_matrix: np.ndarray):
    regrets = regrets_lower_bound_matrix(utils, epsilon_matrix)
    return (np.amax(np.maximum(0, regrets), axis=-1) == 0).astype(int)


def global_sampling(empirical_game: EmpiricalGame, c: float, delta: float, target_epsilon: float, batch_size: int,
                    initial_batch_size: int = None, max_iterations: int = 10000, use_hoeffding=True,
                    streaming_object=None,
                    show_graphs_every: int = -1, verbose=0):
    assert target_epsilon >= 0, 'Epsilon threshold must be greater than or equal to 0'
    assert 0 < delta < 1, 'Failure probability delta must be between 0 and 1'

    if target_epsilon == 0 and verbose >= 1:
        print('WARNING: Setting epsilon threshold to 0 will result in GS running for max_iterations.')

    sample_history = []
    supremum_epsilon = []

    if initial_batch_size is None:
        initial_batch_size = batch_size

    stats = None
    epsilon = None
    for i in range(max_iterations):
        sample_utils = np.array(
            [empirical_game.sample_utils() for _ in range(batch_size if i > 0 else initial_batch_size)])
        if len(sample_history) > 0:
            sample_history.append(sample_history[-1] + batch_size)
        else:
            sample_history.append(initial_batch_size)
        stats, streaming_object = EGTA.get_stats(sample_utils, streaming=True, streaming_object=streaming_object)

        bennett_non_uniform = EGTA.bennet_non_uniform(c, delta, stats)
        if use_hoeffding:
            hoeffding = EGTA.hoeffding_bound(c, delta, stats)
            epsilon = np.minimum(bennett_non_uniform, hoeffding)
        else:
            epsilon = bennett_non_uniform
        supremum_epsilon.append(np.amax(epsilon))

        # print(f'Total Samples: {initial_batch_size + i * batch_size}')

        if supremum_epsilon[-1] <= target_epsilon:
            results = {
                'supremum_epsilon': supremum_epsilon,
                'final_empirical_utils': stats['sample_mean'],
                'final_epsilon_matrix': epsilon,
                'streaming_object': streaming_object
            }
            return sample_history, results

        if show_graphs_every > 0 and (i + 1) % show_graphs_every == 0:
            plt.plot(sample_history, supremum_epsilon)
            plt.axhline(y=target_epsilon, xmax=sample_history[-1])
            plt.title('Global Sampling')
            plt.show()

    if stats is None:
        raise Exception('Stats is None. Weird')
    if epsilon is None:
        raise Exception('Epsilon is None. Weird')

    results = {
        'supremum_epsilon': supremum_epsilon,
        'final_empirical_utils': stats['sample_mean'],
        'final_epsilon_matrix': epsilon,
        'streaming_object': streaming_object
    }
    return sample_history, results


def gs_h_sample_complexity(c: float, delta: float, target_epsilon: float, game_size: int):
    return c ** 2 * (np.log(2) + np.log(game_size) - np.log(delta)) / (2 * target_epsilon ** 2)


def gs_b_sample_complexity(c: float, delta: float, target_epsilon: float, variance: Union[float, np.ndarray],
                           game_size: int):
    if variance == 0:
        return 1
    return (np.log(2) + np.log(game_size) - np.log(delta)) * (c ** 2 / (variance * h(c * target_epsilon / variance)))


def gs_eb_sample_complexity(c: float, delta: float, target_epsilon: float, variance: Union[float, np.ndarray],
                            game_size: int):
    log_term = np.log(3) + np.log(game_size) - np.log(delta)
    return 1 + (8.0 / 3 + np.sqrt(
        4 + 2 / log_term)) * c * log_term / target_epsilon + 2 * variance * log_term / target_epsilon ** 2


def alpha_value_old(c: float, target_epsilon: float, delta: float, game_size: int, num_iterations: int):
    return 2 * c / (3 * target_epsilon) * np.log(3 * num_iterations * game_size / delta)


def alpha_value(c: float, target_epsilon: float, delta: float, game_size: int, num_iterations: int):
    lmbda = 1 / 3.0 + np.sqrt((4 + 2 * np.sqrt(3)) / 3)
    return lmbda * c / target_epsilon * np.log(3 * num_iterations * game_size / delta)


def schedule_length_old(c: float, target_epsilon: float, beta: float):
    return np.ceil(np.log(3 * c / (4 * target_epsilon)) / np.log(beta)).astype('int')


def schedule_length(c: float, target_epsilon: float, beta: float):
    lmbda = 1 / 3.0 + np.sqrt((4 + 2 * np.sqrt(3)) / 3)
    return max(1, np.ceil(np.log(c / (2 * lmbda * target_epsilon)) / np.log(beta)).astype('int'))


def schedule_length_regret(c: float, target_epsilon: float, beta: float):
    kappa = 1.0 / 3 + np.sqrt(4.0 / 3 + np.sqrt(4.0 / 3))
    return max(1, math.ceil(np.log(c ** 2 / (4 * kappa * target_epsilon ** 2)) / np.log(beta)))


def ps_sample_complexity(c: float, delta: float, target_epsilon: float, variance: Union[float, np.ndarray], beta: float,
                         game_size: int):
    T = schedule_length(c, target_epsilon, beta)
    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    kappa = 4.0 / 3 + np.sqrt(1 + 1 / (2 * log_term))
    return 1 + beta * (kappa * c * log_term / target_epsilon + variance * log_term / target_epsilon ** 2 +
                       np.sqrt(kappa * c * log_term / target_epsilon * variance * log_term / target_epsilon ** 2 +
                               (variance * log_term / target_epsilon ** 2) ** 2))


def ps_query_complexity(c: float, delta: float, target_epsilon: float, variance: np.ndarray, beta: float,
                        game_size: int):
    variance = np.amax(variance, axis=-1)
    return np.sum(ps_sample_complexity(c, delta, target_epsilon, variance, beta, game_size))


def ps_regret_pure_query_complexity(c: float, delta: float, target_epsilon: float, utils: np.ndarray, gamma: float,
                                    variance: np.ndarray, beta: float,
                                    game_size: int):
    T = schedule_length_regret(c, target_epsilon, beta)
    num_players = len(utils.shape) - 1
    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    reg_matrix = regrets_matrix(utils)
    max_adjacent_variance = np.empty_like(variance)
    for i in range(num_players):
        max_adjacent_variance[..., i] = np.amax(variance[..., i], axis=i, keepdims=True)
    reg_matrix_adjusted = np.maximum(reg_matrix - gamma, 0.001)
    regret_complexities = 2 + 2 * beta * log_term * (
                10 * c / reg_matrix_adjusted + 16 * max_adjacent_variance / np.square(reg_matrix_adjusted))
    we_complexities = 2 + 2 * beta * log_term * (5 * c / (2 * target_epsilon) + variance / (target_epsilon ** 2))
    queries = np.minimum(regret_complexities, we_complexities)
    queries = np.sum(np.amax(queries, axis=-1))
    return queries


def ps_regret_mixed_query_complexity(c: float, delta: float, target_epsilon: float, utils: np.ndarray,
                                     variance: np.ndarray, beta: float,
                                     game_size: int):
    T = schedule_length_regret(c, target_epsilon, beta)
    num_players = len(utils.shape) - 1
    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    reg_matrix = regrets_matrix(utils)
    max_adjacent_variance = np.empty_like(variance)
    for i in range(num_players):
        max_adjacent_variance[..., i] = np.amax(variance[..., i], axis=i, keepdims=True)
    reg_matrix_adjusted = np.maximum(reg_matrix - target_epsilon, 0.001)
    regret_complexities = 2 + 2 * beta * log_term * (
                12.5 * c / reg_matrix_adjusted + 25 * max_adjacent_variance / np.square(reg_matrix_adjusted))
    we_complexities = 2 + 2 * beta * log_term * (5 * c / (2 * target_epsilon) + variance / (target_epsilon ** 2))
    queries = np.minimum(regret_complexities, we_complexities)
    queries = np.sum(np.amax(queries, axis=-1))
    return queries


def progressive_sampling_with_pruning(empirical_game: EmpiricalGame, c: float, delta: float,
                                      target_epsilon: float, beta: float,
                                      well_estimated_pruning: bool = True, regret_pruning: bool = True,
                                      wimpy_variance: bool = False,
                                      max_iterations: int = 1000,
                                      return_intermediate_utils: bool = False,
                                      return_epsilon_matrices: bool = False,
                                      old_sampling_schedule: bool = False,
                                      old_regret_pruning: bool = False,
                                      count_each_pruning_contribution: bool = False,
                                      show_graphs_every: int = -1,
                                      verbose=2):
    assert target_epsilon > 0, 'Epsilon threshold must be greater than 0'
    assert beta > 1, 'Geometric ratio beta must be greater than 1'
    assert 0 < delta < 1, 'Failure probability delta must be between 0 and 1'

    utils_shape = empirical_game.get_utils_shape()
    game_size = np.prod(utils_shape)
    num_players = utils_shape[-1]

    m = 0
    if old_sampling_schedule:
        T = schedule_length_old(c, target_epsilon, beta)
        alpha = alpha_value_old(c, target_epsilon, delta, game_size, T)
    else:
        T = schedule_length(c, target_epsilon, beta)
        alpha = alpha_value(c, target_epsilon, delta, game_size, T)

    final_empirical_utils = np.empty(utils_shape)
    intermediate_empirical_utils = []
    active_utils = np.full(utils_shape, True)
    num_active_utils = [game_size]
    num_active_profiles = [game_size / num_players]
    regret_active_utils = None
    well_estimated_active_utils = None
    regret_num_active_utils = None
    well_estimated_num_active_utils = None
    regret_num_active_profiles = None
    well_estimated_num_active_profiles = None
    if count_each_pruning_contribution:
        assert regret_pruning
        assert well_estimated_pruning
        regret_active_utils = np.full(utils_shape, True)
        well_estimated_active_utils = np.full(utils_shape, True)
        regret_num_active_utils = [game_size]
        well_estimated_num_active_utils = [game_size]
        regret_num_active_profiles = [game_size / num_players]
        well_estimated_num_active_profiles = [game_size / num_players]
    sample_history = []
    supremum_epsilon = []
    infimum_epsilon = []
    epsilon_matrices = []

    streaming_object = None
    stats = None
    for t in range(1, min(T + 1, max_iterations + 1)):
        alpha_beta_t = np.ceil(alpha * np.power(beta, t)).astype(int)
        m_prime = alpha_beta_t - m
        if verbose >= 2:
            print(m_prime)
            print(t)
        m = alpha_beta_t

        sample_utils = np.array([empirical_game.sample_utils() for _ in range(m_prime)])
        if len(sample_history) > 0:
            sample_history.append(sample_history[-1] + m_prime)
        else:
            sample_history.append(m_prime)
        stats, streaming_object = EGTA.get_stats(sample_utils, streaming=True, streaming_object=streaming_object,
                                                 rademacher=False)

        hoeffding = EGTA.hoeffding_bound(c, delta, stats, T)
        if wimpy_variance:
            bennett = EGTA.bennet_union_bound(c, delta, stats, T, return_matrix=True)
        else:
            bennett = EGTA.bennet_non_uniform(c, delta, stats, T)
        epsilon = np.minimum(bennett, hoeffding)
        supremum_epsilon.append(np.amax(epsilon))
        infimum_epsilon.append(np.amin(epsilon))

        if return_epsilon_matrices:
            epsilon_matrices.append(epsilon)

        if return_intermediate_utils:
            if len(intermediate_empirical_utils) > 0:
                intermediate_empirical_utils.append(np.copy(intermediate_empirical_utils[-1]))
                intermediate_empirical_utils[-1][active_utils] = stats['sample_mean'][active_utils]
            else:
                intermediate_empirical_utils.append(stats['sample_mean'])

        epsilon_sensitive_regrets = None
        num_active_utils.append(num_active_utils[-1])
        num_active_profiles.append(num_active_profiles[-1])
        if well_estimated_pruning:
            newly_pruned_utils = np.logical_and(active_utils, epsilon <= target_epsilon)
            final_empirical_utils[newly_pruned_utils] = stats['sample_mean'][newly_pruned_utils]
            active_utils[newly_pruned_utils] = False
            num_active_utils[-1] = np.count_nonzero(active_utils)
            num_active_profiles[-1] = np.count_nonzero(np.amax(active_utils, axis=-1))
            if verbose >= 2:
                print(f'After Well Estimated: {num_active_utils[-1]} Indices, {num_active_profiles[-1]} Profiles')
        if regret_pruning:
            if old_regret_pruning:
                epsilon_sensitive_regrets = regrets_lower_bound_matrix(stats['sample_mean'],
                                                                       epsilon=supremum_epsilon[-1])
            else:
                epsilon_sensitive_regrets = regrets_lower_bound_matrix(stats['sample_mean'], epsilon_matrix=epsilon)
            newly_pruned_utils = np.logical_and(active_utils, epsilon_sensitive_regrets > 0)
            final_empirical_utils[newly_pruned_utils] = stats['sample_mean'][newly_pruned_utils]
            active_utils[newly_pruned_utils] = False
            num_active_utils[-1] = np.count_nonzero(active_utils)
            num_active_profiles[-1] = np.count_nonzero(np.amax(active_utils, axis=-1))
            if verbose >= 2:
                print(f'After Regret: {num_active_utils[-1]} Indices, {num_active_profiles[-1]} Profiles')
        if count_each_pruning_contribution:
            if epsilon_sensitive_regrets is None:
                print('PROBLEM: epsilon_sensitive_regrets should not be None')
                exit()
            we_newly_pruned_utils = np.logical_and(well_estimated_active_utils, epsilon <= target_epsilon)
            reg_newly_pruned_utils = np.logical_and(regret_active_utils, epsilon_sensitive_regrets > 0)
            well_estimated_active_utils[we_newly_pruned_utils] = False
            regret_active_utils[reg_newly_pruned_utils] = False
            well_estimated_num_active_utils.append(np.count_nonzero(well_estimated_active_utils))
            regret_num_active_utils.append(np.count_nonzero(regret_active_utils))
            well_estimated_num_active_profiles.append(np.count_nonzero(np.amax(well_estimated_active_utils, axis=-1)))
            regret_num_active_profiles.append(np.count_nonzero(np.amax(regret_active_utils, axis=-1)))
        if show_graphs_every > 0 and (t + 1) % show_graphs_every == 0:
            plt.plot(sample_history, supremum_epsilon)
            plt.axhline(y=target_epsilon, xmax=sample_history[-1])
            plt.title('Progressive Sampling with Pruning')
            plt.show()
        if num_active_utils[-1] == 0 or (not well_estimated_pruning and supremum_epsilon[-1] <= target_epsilon):
            if verbose >= 1:
                print(f'Achieved Target Epsilon on Iteration {t}')
            results = {
                'supremum_epsilon': supremum_epsilon,
                'infimum_epsilon': infimum_epsilon,
                'num_active_utils': num_active_utils,
                'num_active_profiles': num_active_profiles,
                'regret_active_utils': regret_num_active_utils,
                'well_estimated_active_utils': well_estimated_num_active_utils,
                'regret_active_profiles': regret_num_active_profiles,
                'well_estimated_active_profiles': well_estimated_num_active_profiles,
                'final_empirical_utilities': final_empirical_utils,
                'empirical_utils': intermediate_empirical_utils if return_intermediate_utils else None,
                'epsilon_matrices': epsilon_matrices if return_epsilon_matrices else None
            }
            return sample_history, results

    if stats is None:
        raise Exception('Seems that T or max_iterations is 0. No iterations of PSP run')

    print('Did not meet Target Epsilon')

    final_empirical_utils[active_utils] = stats['sample_mean'][active_utils]
    results = {
        'supremum_epsilon': supremum_epsilon,
        'infimum_epsilon': infimum_epsilon,
        'num_active_utils': num_active_utils,
        'num_active_profiles': num_active_profiles,
        'regret_active_utils': regret_num_active_utils,
        'well_estimated_active_utils': well_estimated_num_active_utils,
        'regret_active_profiles': regret_num_active_profiles,
        'well_estimated_active_profiles': well_estimated_num_active_profiles,
        'final_empirical_utilities': final_empirical_utils,
        'empirical_utils': intermediate_empirical_utils if return_intermediate_utils else None,
        'epsilon_matrices': epsilon_matrices if return_epsilon_matrices else None
    }
    return sample_history, results


def sampling_schedule_well_estimated(c: float, target_epsilon: float, delta: float, beta: float, game_size: int):
    kappa = 1.0 / 3 + np.sqrt(4.0 / 3 + np.sqrt(4.0 / 3))
    T = max(1, math.ceil(np.log(c / (2 * target_epsilon * kappa)) / np.log(beta)))
    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    alpha = c * log_term * kappa / target_epsilon
    schedule = [math.ceil(alpha * beta)]
    for t in range(2, T + 1):
        schedule.append(math.ceil(alpha * np.power(beta, t)) - schedule[-1])
    return schedule


def sampling_schedule_regret(c: float, target_epsilon: float, delta: float, beta: float, game_size: int):
    if target_epsilon >= c / 2:
        return sampling_schedule_well_estimated(c, target_epsilon, delta, beta, game_size)
    kappa = 1.0 / 3 + np.sqrt(4.0 / 3 + np.sqrt(4.0 / 3))
    T = max(1, math.ceil(np.log(c / (2 * target_epsilon * kappa)) / np.log(beta)))
    k = 1.5
    log_term = np.log(3) + np.log(game_size) + np.log(T + math.ceil((k - 1) * T)) - np.log(delta)
    alpha = 2 * log_term * kappa
    alpha_p = c * log_term * kappa / target_epsilon
    T_linear = math.ceil((k - 1) * T)

    increment = (alpha_p - alpha) / T_linear
    cumulative_schedule = []
    for t in range(1, T_linear + 1):
        cumulative_schedule.append(math.ceil(alpha + increment * t))
    for t in range(1, T + 1):
        cumulative_schedule.append(math.ceil(alpha_p * np.power(beta, t)))
    # print(cumulative_schedule)
    # print(c * c * log_term / (2 * target_epsilon ** 2))
    schedule = [cumulative_schedule[0]]
    for i in range(1, len(cumulative_schedule)):
        schedule.append(cumulative_schedule[i] - cumulative_schedule[i - 1])
    return schedule


def sampling_schedule_regret_geometric(c: float, target_epsilon: float, delta: float, beta: float, game_size: int):
    kappa = 1.0 / 3 + np.sqrt(4.0 / 3 + np.sqrt(4.0 / 3))
    T = max(1, math.ceil(np.log(c ** 2 / (4 * kappa * target_epsilon ** 2)) / np.log(beta)))
    log_term = np.log(3) + np.log(game_size) + np.log(T) - np.log(delta)
    alpha = 2 * log_term * kappa
    schedule = [math.ceil(alpha * beta)]
    for t in range(2, T + 1):
        schedule.append(math.ceil(alpha * np.power(beta, t)) - schedule[-1])
    return schedule


def well_estimated_pruning_criteria(empirical_utilities: np.ndarray, epsilon_bounds: np.ndarray,
                                    target_epsilon: float):
    return epsilon_bounds <= target_epsilon


def regret_pruning_old_criteria_plus(empirical_utilities: np.ndarray, epsilon_bounds: np.ndarray,
                                     target_epsilon: float):
    epsilon_sensitive_regrets = regrets_lower_bound_matrix(empirical_utilities, epsilon_matrix=epsilon_bounds)
    return epsilon_sensitive_regrets > 0


def regret_pruning_plus(empirical_utilities: np.ndarray, epsilon_bounds: np.ndarray,
                        target_epsilon: float):
    epsilon_sensitive_regrets = regrets_lower_bound_matrix(empirical_utilities, epsilon_matrix=epsilon_bounds)
    return epsilon_sensitive_regrets > np.maximum(0, 3 * target_epsilon - epsilon_bounds)


def regret_pruning_mixed(empirical_utilities: np.ndarray, epsilon_bounds: np.ndarray,
                         target_epsilon: float):
    epsilon_sensitive_regrets = regrets_lower_bound_matrix(empirical_utilities, epsilon_matrix=epsilon_bounds)
    return epsilon_sensitive_regrets > target_epsilon + epsilon_bounds


def generalized_progressive_sampling(empirical_game: EmpiricalGame, c: float, delta: float,
                                     target_epsilon: float,
                                     sampling_schedule: List[int],
                                     pruning_criteria: List[Callable[[np.ndarray, np.ndarray, float],
                                                                     np.ndarray]],
                                     wimpy_variance: bool = False,
                                     max_iterations: int = 1000,
                                     return_intermediate_utils: bool = False,
                                     return_epsilon_matrices: bool = False,
                                     pruned_utility_matrix=None,
                                     pruned_epsilon_matrix=None,
                                     two_player_zero_sum=False,
                                     show_graphs_every: int = -1,
                                     verbose=2):
    assert target_epsilon > 0, 'Epsilon threshold must be greater than 0'
    assert 0 < delta < 1, 'Failure probability delta must be between 0 and 1'

    utils_shape = empirical_game.get_utils_shape()
    game_size = np.prod(utils_shape)
    num_players = utils_shape[-1]

    m = 0
    T = len(sampling_schedule)

    if pruned_epsilon_matrix is not None and pruned_utility_matrix is not None:
        final_empirical_utils = pruned_utility_matrix
        final_epsilon_matrix = pruned_epsilon_matrix
        active_utils = (pruned_epsilon_matrix == 0)
    else:
        final_empirical_utils = np.empty(utils_shape)
        final_epsilon_matrix = np.empty(utils_shape)
        active_utils = np.full(utils_shape, True)
    intermediate_empirical_utils = []
    num_active_utils = [np.count_nonzero(active_utils)]
    num_active_profiles = [np.count_nonzero(np.amax(active_utils, axis=-1))]

    sample_history = []
    supremum_epsilon = []
    infimum_epsilon = []
    epsilon_matrices = []

    streaming_object = None
    stats = None
    for t in range(1, min(T + 1, max_iterations + 1)):
        m_prime = sampling_schedule[t - 1]
        if verbose >= 2:
            print(f'Iteration {t}, Drawing {m_prime} new samples')
        m += m_prime

        sample_utils = np.array([empirical_game.sample_utils() for _ in range(m_prime)])
        sample_history.append(m)
        stats, streaming_object = EGTA.get_stats(sample_utils, streaming=True, streaming_object=streaming_object,
                                                 rademacher=False)

        hoeffding = EGTA.hoeffding_bound(c, delta, stats, T)
        if wimpy_variance:
            bennett = EGTA.bennet_union_bound(c, delta, stats, T, return_matrix=True)
        else:
            bennett = EGTA.bennet_non_uniform(c, delta, stats, T)
        epsilon = np.minimum(bennett, hoeffding)

        if two_player_zero_sum:
            active_profiles = np.amax(active_utils, axis=-1)
            final_empirical_utils[active_profiles] = stats['sample_mean'][active_profiles]
            final_epsilon_matrix[active_profiles] = epsilon[active_profiles]
            final_empirical_utils[..., 1] = -final_empirical_utils[..., 0]
            final_epsilon_matrix[..., 1] = final_epsilon_matrix[..., 0]
        else:
            final_empirical_utils[active_utils] = stats['sample_mean'][active_utils]
            final_epsilon_matrix[active_utils] = epsilon[active_utils]
        if return_intermediate_utils:
            intermediate_empirical_utils.append(np.copy(final_empirical_utils))
        if return_epsilon_matrices:
            epsilon_matrices.append(np.copy(final_epsilon_matrix))

        num_active_utils.append(num_active_utils[-1])
        num_active_profiles.append(num_active_profiles[-1])
        for criteria in pruning_criteria:
            newly_pruned_utils = np.logical_and(active_utils,
                                                criteria(final_empirical_utils, final_epsilon_matrix, target_epsilon))
            active_utils[newly_pruned_utils] = False
            num_active_utils[-1] = np.count_nonzero(active_utils)
            num_active_profiles[-1] = np.count_nonzero(np.amax(active_utils, axis=-1))
            if verbose >= 2:
                print(
                    f' After Pruning via {criteria.__name__}: {num_active_utils[-1]} Active Indices, {num_active_profiles[-1]} Active Profiles')
        if show_graphs_every > 0 and (t + 1) % show_graphs_every == 0:
            plt.plot(sample_history, supremum_epsilon)
            plt.axhline(y=target_epsilon, xmax=sample_history[-1])
            plt.title('Progressive Sampling with Pruning')
            plt.show()
        if num_active_utils[-1] == 0:
            if verbose >= 1:
                print(f'Achieved Target Epsilon on Iteration {t}')
            results = {
                'supremum_epsilon': supremum_epsilon,
                'infimum_epsilon': infimum_epsilon,
                'num_active_utils': num_active_utils,
                'num_active_profiles': num_active_profiles,
                'final_empirical_utilities': final_empirical_utils,
                'final_epsilon_matrix': final_epsilon_matrix,
                'empirical_utils': intermediate_empirical_utils if return_intermediate_utils else None,
                'epsilon_matrices': epsilon_matrices if return_epsilon_matrices else None
            }
            return sample_history, results

    if stats is None:
        raise Exception('Seems that T or max_iterations is 0. No iterations of PSP run')

    print('Did not meet Target Epsilon')

    final_empirical_utils[active_utils] = stats['sample_mean'][active_utils]
    results = {
        'supremum_epsilon': supremum_epsilon,
        'infimum_epsilon': infimum_epsilon,
        'num_active_utils': num_active_utils,
        'num_active_profiles': num_active_profiles,
        'final_empirical_utilities': final_empirical_utils,
        'final_epsilon_matrix': final_epsilon_matrix,
        'empirical_utils': intermediate_empirical_utils if return_intermediate_utils else None,
        'epsilon_matrices': epsilon_matrices if return_epsilon_matrices else None
    }
    return sample_history, results


def progressive_sampling(empirical_game: EmpiricalGame, c: float, delta: float,
                         target_epsilon: float, beta: float,
                         max_iterations: int = 1000,
                         return_intermediate_utils: bool = False,
                         return_epsilon_matrices: bool = False,
                         show_graphs_every: int = -1,
                         verbose=2):
    # Do progressive_sampling_with_pruning without pruning
    sample_history, results = progressive_sampling_with_pruning(empirical_game, c, delta, target_epsilon, beta,
                                                                well_estimated_pruning=False,
                                                                regret_pruning=False,
                                                                max_iterations=max_iterations,
                                                                return_intermediate_utils=return_intermediate_utils,
                                                                return_epsilon_matrices=return_epsilon_matrices,
                                                                show_graphs_every=show_graphs_every, verbose=verbose)
    results.pop('num_active_profiles')
    return sample_history, results


def regret_complexity_alpha(c, reg, v, v_max_adj):
    cr = c * reg
    v_cr = 2 * v / (5 * cr)
    v_adj_cr = 2 * v_max_adj / (5 * cr)

    a_0 = v_cr
    a_1 = 0.5 - 2 * v_cr
    a_2 = v_cr - v_adj_cr - 1.5

    q = a_1 / 3 - np.power(a_2, 2) / 9
    r = (a_1 * a_2 - 3 * a_0) / 6 - np.power(a_2, 3) / 27

    theta = np.arccos(r / np.power(-q, 1.5))
    phi = theta / 3 - 2 * np.pi / 3

    return 2 * np.sqrt(-q) * np.cos(phi) - a_2 / 3


def maximum_adjacent_var(variances: np.ndarray, opponents_profile: List, player: int, player_actions: List[int] = None):
    opponents_profile.append(player)
    if player_actions is not None:
        opponents_profile[player] = player_actions
    else:
        opponents_profile[player] = slice(variances.shape[player])
    variances = variances[tuple(opponents_profile)]
    indices = np.argpartition(variances, -2)[-2:]
    second_idx, top_idx = indices[np.argsort(variances[indices])]
    second_var, top_var = variances[second_idx], variances[top_idx]
    max_adj_var = np.full(variances.shape, top_var)
    max_adj_var[top_idx] = second_var
    return max_adj_var


def maximum_adjacent_var_player_matrix(variances: np.ndarray, player: int):
    variances = variances[..., player]
    shape = variances.shape
    dims_expand = tuple(i for i in range(len(shape)) if i != player)
    indices = np.take_along_axis(np.argpartition(variances, -2, axis=player),
                                 np.expand_dims(np.array([-2, -1]), axis=dims_expand),
                                 axis=player)
    sorted_indices = np.take_along_axis(indices,
                                        np.argsort(np.take_along_axis(variances, indices, axis=player), axis=player),
                                        axis=player)
    top_indices = np.take_along_axis(sorted_indices,
                                     np.expand_dims(np.array([-1]), axis=dims_expand),
                                     axis=player)
    second_indices = np.take_along_axis(sorted_indices,
                                        np.expand_dims(np.array([-2]), axis=dims_expand),
                                        axis=player)
    top_var = np.take_along_axis(variances, top_indices, axis=player)
    second_var = np.take_along_axis(variances, second_indices, axis=player)
    reps = np.ones(len(shape), dtype=np.int8)
    reps[player] = shape[player]
    max_adj_vars = np.tile(top_var, reps)
    np.put_along_axis(max_adj_vars, top_indices, second_var, axis=player)
    return max_adj_vars


def maximum_adjacent_var_matrix(variances: np.ndarray):
    num_players = variances.shape[-1]
    max_adj_vars = np.empty(variances.shape)
    for player in range(num_players):
        max_adj_vars[..., player] = maximum_adjacent_var_player_matrix(variances, player)
    return max_adj_vars


def regret_pruning_complexities(c: float, target_epsilon: float, utils: np.ndarray, variances: np.ndarray,
                                delta: float = 0.05, beta: float = 1.5, use_alphas=True):
    assert utils.shape == variances.shape
    max_adj_vars = maximum_adjacent_var_matrix(variances)
    T = schedule_length(c, target_epsilon, beta)
    game_size = np.prod(utils.shape)
    log_term = np.log(3) + np.log(T) + np.log(game_size) - np.log(delta)
    regrets = regrets_matrix(utils)
    zero_regret_indices = (regrets == 0)
    regrets[zero_regret_indices] = c / 100
    if use_alphas:
        alphas = regret_complexity_alpha(c, regrets, variances, max_adj_vars)
        complexities = 1 + 2 * beta * log_term * (
                2.5 * c / (alphas * regrets) + 4 * variances / np.power(alphas * regrets, 2))
    else:
        alphas = np.full(regrets.shape, 0.5)
        complexities = np.maximum(
            1 + 2 * beta * log_term * (2.5 * c / (alphas * regrets) + 4 * variances / np.power(alphas * regrets, 2)),
            1 + 2 * beta * log_term * (
                    2.5 * c / ((1 - alphas) * regrets) + 4 * max_adj_vars / np.power((1 - alphas) * regrets, 2))
        )
    complexities[zero_regret_indices] = np.inf
    return complexities


def iterated_dominance(utils: np.ndarray):
    num_strategies = utils.shape[:-1]
    num_players = utils.shape[-1]
    print(num_strategies)
    print(num_players)

    eliminated = [[] for _ in range(num_players)]

    removed = True
    while removed:
        removed = False
        for p in range(num_players):
            for s1, s2 in itertools.combinations(range(num_strategies[p]), 2):
                idx_1 = [slice(num_strategies[i]) for i in range(num_players)]
                idx_2 = [slice(num_strategies[i]) for i in range(num_players)]
                idx_1[p] = s1
                idx_2[p] = s2
                idx_1.append(p)
                idx_2.append(p)
                idx_1 = tuple(idx_1)
                idx_2 = tuple(idx_2)
                dominance_1 = (utils[idx_1] >= utils[idx_2])
                dominance_2 = (utils[idx_2] >= utils[idx_1])
                print(dominance_1)
                print(dominance_2)
                for strategy in eliminated[p]:
                    dominance_1[strategy] = True
                    dominance_2[strategy] = True
                if np.all(dominance_1):
                    eliminated[p].append(idx_1)
                    removed = True
                elif np.all(dominance_2):
                    eliminated[p].append(idx_2)
                    removed = True

    return eliminated
