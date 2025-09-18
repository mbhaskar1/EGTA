from objGeneration import random

from NoisyGame import Noise, NoiseType, NoiseFactor, NoisyGame
from GamutGame import GamutGame
import scipy.stats
from typing import List
import numpy as np


def construct_empirical_game(game, uniform_noise=0, agent_noise=0, agent_action_noise=0,
                             complete_noise=100, noise_distribution=scipy.stats.uniform, noise_args=None,
                             noise_multiplier_distribution=None, noise_multiplier_args=None):
    utils_shape = tuple([*game.get_num_actions(), game.get_players()])
    print(utils_shape)
    if noise_multiplier_distribution is not None:
        if noise_multiplier_args is None:
            noise_multiplier_args = {'loc': 0, 'scale': 1}
        complete_noise_multipliers = noise_multiplier_distribution.rvs(size=utils_shape, **noise_multiplier_args)
    else:
        complete_noise_multipliers = None

    if noise_args is None:
        noise_args = {'loc': 0, 'scale': 1}

    noise_sources = []
    if uniform_noise > 0:
        noise_sources.append(Noise(NoiseFactor.uniform, NoiseType.additive, noise_distribution,
                                   noise_args, shift=-uniform_noise, scale=2 * uniform_noise,
                                   noise_multipliers=complete_noise_multipliers))
    if agent_noise > 0:
        noise_sources.append(Noise(NoiseFactor.agent, NoiseType.additive, noise_distribution,
                                   noise_args, shift=-agent_noise, scale=2 * agent_noise,
                                   noise_multipliers=complete_noise_multipliers))
    if agent_action_noise > 0:
        noise_sources.append(Noise(NoiseFactor.agent_action, NoiseType.additive, noise_distribution,
                                   noise_args, shift=-agent_action_noise, scale=2 * agent_action_noise,
                                   noise_multipliers=complete_noise_multipliers))
    if complete_noise > 0:
        noise_sources.append(Noise(NoiseFactor.complete, NoiseType.additive, noise_distribution,
                                   noise_args, shift=-complete_noise, scale=2 * complete_noise,
                                   noise_multipliers=complete_noise_multipliers))

    return NoisyGame(game, [], noise_sources)


def congestion_facilities_to_action(facilities: List[int]):
    assert len(facilities) > 0
    assert min(facilities) >= 0

    m = max(facilities)
    bin_rep = ['0'] * (m + 1)
    for facility in facilities:
        bin_rep[m - facility] = '1'
    return int(''.join(bin_rep), 2) - 1


def grab_the_dollar(actions, c=200):
    return GamutGame('GrabTheDollar', c,
                     '-actions', f'{actions}',
                     '-random_params',
                     '-normalize',
                     '-max_payoff', f'{int(c / 2)}',
                     '-min_payoff', f'{int(-c / 2)}')


def travelers_dilemma(players, actions, reward=None, c=200):
    params = [
        '-players', f'{players}',
        '-actions', f'{actions}',
        '-random_params',
        '-normalize',
        '-max_payoff', f'{int(c / 2)}',
        '-min_payoff', f'{int(-c / 2)}'
    ]
    if reward is not None:
        params.append('-reward')
        params.append(f'{reward}')
    return GamutGame('TravelersDilemma', c, *params)


def war_of_attrition(actions, val_low=None, val_high=None, dec_low=None, dec_high=None, c=200):
    params = [
        '-actions', f'{actions}',
        '-random_params',
        '-normalize',
        '-max_payoff', f'{int(c / 2)}',
        '-min_payoff', f'{int(-c / 2)}'
    ]
    if val_low is not None:
        params.append('-valuation_low')
        params.append(f'{val_low}')
    if val_high is not None:
        params.append('-valuation_high')
        params.append(f'{val_high}')
    if dec_low is not None:
        params.append('-decrement_low')
        params.append(f'{dec_low}')
    if dec_high is not None:
        params.append('-decrement_high')
        params.append(f'{dec_high}')
    return GamutGame('WarOfAttrition', c, *params)


def bertrand_oligopoly(players, actions, c=200):
    return GamutGame('BertrandOligopoly', c,
                     '-players', f'{players}',
                     '-actions', f'{actions}',
                     '-random_params',
                     '-normalize',
                     '-max_payoff', f'{int(c / 2)}',
                     '-min_payoff', f'{int(-c / 2)}')


def congestion_game(players, facilities, c=200):
    return GamutGame('CongestionGame', c,
                     '-players', f'{players}',
                     '-facilities', f'{facilities}',
                     '-random_params',
                     '-normalize',
                     '-max_payoff', f'{int(c / 2)}',
                     '-min_payoff', f'{int(-c / 2)}',
                     '-sym_funcs', '0')


def random_zero_sum(actions, c=200):
    return GamutGame('RandomZeroSum', c,
                     '-actions', f'{actions}',
                     '-random_params',
                     '-normalize',
                     '-max_payoff', f'{int(c / 2)}',
                     '-min_payoff', f'{int(-c / 2)}')


def covariant_game(players, actions, r=None, c=200):
    params = [
        '-players', f'{players}',
        '-actions', f'{actions}',
        '-random_params',
        '-normalize',
        '-max_payoff', f'{int(c / 2)}',
        '-min_payoff', f'{int(-c / 2)}'
    ]
    if r is not None:
        params.append('-r')
        params.append(f'{r}')
    return GamutGame('CovariantGame', c, *params)


def dispersion_game(players, actions, c=200):
    return GamutGame('DispersionGame', c,
                     '-players', f'{players}',
                     '-actions', f'{actions}',
                     '-random_params',
                     '-normalize',
                     '-max_payoff', f'{int(c / 2)}',
                     '-min_payoff', f'{int(-c / 2)}')


def congestion_game_restricted(players, facilities, num_actions, c=200):
    game = congestion_game(players, facilities, c)
    indices = []
    for player in range(players):
        indices.append(random.sample(list(range(2 ** facilities - 1)), num_actions))
    ix_grid = np.ix_(*indices)
    game.utils = game.utils[ix_grid]
    game.num_actions = [num_actions] * players
    return game


def custom_game(utils, c=200):
    return GamutGame('CustomGame', c, utils)
