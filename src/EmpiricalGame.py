import itertools
import numpy as np
from typing import List, Optional
import enum

import scipy.stats

from DiscreteDistribution import DiscreteDistribution
from GamutGame import GamutGame, Action


class ChancePlayer:
    """Helper Class for representing Chance Players. Look at EmpiricalGame class for more info."""

    def __init__(self, player_number: int, distribution: DiscreteDistribution):
        """Initializes ChancePlayer object

        :param player_number: Player index of chance player
        :param distribution: DiscreteDistribution object
        """
        self.player_number = player_number
        self.distribution = distribution


class NoiseFactor(enum.Enum):
    uniform = 1
    agent = 2
    agent_action = 3
    profile = 4
    complete = 5


class NoiseType(enum.Enum):
    additive = 1
    multiplicative = 2


class Noise:
    def __init__(self, factor: NoiseFactor, noise_type: NoiseType, distribution: scipy.stats.rv_continuous,
                 dist_args: dict, agent_action_selector=0):
        self.factor = factor
        self.noise_type = noise_type
        self.distribution = distribution
        self.dist_args = dist_args
        self.agent_action_selector = agent_action_selector

    # Assumes last dimension is players
    def apply_noise(self, sample_utils: np.ndarray):
        rvs = 1 if self.noise_type == NoiseType.multiplicative else 0
        if self.factor == NoiseFactor.uniform:
            rvs = self.distribution.rvs(size=(), **self.dist_args)
        if self.factor == NoiseFactor.agent:
            rvs = self.distribution.rvs(size=(sample_utils.shape[-1]), **self.dist_args)
        if self.factor == NoiseFactor.agent_action:
            size = tuple(sample_utils.shape[i] if i == self.agent_action_selector else 1 for i in range(
                len(sample_utils.shape)))
            rvs = self.distribution.rvs(size=sample_utils.shape[self.agent_action_selector], **self.dist_args) \
                .reshape(size)
        if self.factor == NoiseFactor.profile:
            rvs = self.distribution.rvs(size=sample_utils.shape[:-1], **self.dist_args) \
                .reshape(sample_utils.shape[:-1] + (1,))
        if self.factor == NoiseFactor.complete:
            rvs = self.distribution.rvs(size=sample_utils.shape, **self.dist_args)
        if self.noise_type == NoiseType.additive:
            sample_utils += rvs
        if self.noise_type == NoiseType.multiplicative:
            sample_utils *= rvs


class EmpiricalGame:
    """Extends a GamutGame by allowing for certain players to be converted to Chance Players

    Chance Players are players that come with a distribution over their possible actions, which can then be sampled.
    These Chance Players can be used to represent external factors and noise that influence utilities in games.
    """

    def __init__(self, game: GamutGame, chance_players: List[ChancePlayer], noise_sources: List[Noise] = None):
        """Initializes EmpiricalGame object

        :param game: GamutGame game object to be extended into an empirical game
        :param chance_players: List of Chance Players to substitute active players in game
        """
        self.game = game
        self.chance_players = sorted(chance_players, key=lambda player: player.player_number)
        self.chance_player_indices = [chance_player.player_number for chance_player in self.chance_players]
        self.player_indices = [player for player in range(game.get_players())
                               if player not in self.chance_player_indices]
        self.noise_sources = noise_sources if noise_sources is not None else []

    def get_utils(self, chance_actions: List[Action], player_actions: Optional[List[Action]] = None,
                  players: Optional[List[int]] = None) -> np.ndarray:
        """Gets the utilities corresponding to certain chance actions being taken. Can also get a restricted section of
        these utilities.

        :param chance_actions: List of chance actions being taken
        :param player_actions: List of player actions to restrict returned utilities (defaults to no restriction by Player Actions)
        :param players: List of player indices to restrict returned utilities (defaults to all active players)
        :return: Numpy array containing utilities corresponding to the provided chance actions, player actions, and players. Look at get_utils in GamutGame for more information.
        """
        actions = chance_actions
        if player_actions is not None:
            actions.extend(player_actions)
        if players is None:
            players = self.player_indices
        return self.game.get_utils(actions=actions, players=players)

    def sample_actions(self) -> List[Action]:
        """Samples chance actions for each chance player from their respective distributions

        :return: List of chance actions
        """
        chance_actions = []
        for chance_player in self.chance_players:
            chance_actions.append(Action(chance_player.player_number, chance_player.distribution.sample()))
        return chance_actions

    def sample_utils(self) -> np.ndarray:
        """Sample chance actions for each chance player and get resulting utilities for each combination of player
        actions

        :return: Numpy array containing utilities corresponding to each combination of player actions and each player. Look at get_utils in GamutGame for more info.
        """
        sample_utils = self.get_utils(self.sample_actions())
        for noise in self.noise_sources:
            noise.apply_noise(sample_utils)
        return sample_utils

    def expected_utils(self):
        """Get expectation over utilities with respect to the Chance Player distributions.

        :return: Numpy array containing expected utilities corresponding to each combination of player actions and each player.
        """
        expected_utils = self.get_utility_matrix()
        if len(self.chance_players) == 0:
            return expected_utils
        probabilities = [np.array(player.distribution.get_probabilities(), np.longdouble)
                         for player in self.chance_players]
        for prob_arr in probabilities:
            expected_utils = np.tensordot(expected_utils, prob_arr, ([0], [0]))
        return expected_utils

    def variance_utils(self):
        utils = self.get_utility_matrix()
        if len(self.chance_players) == 0:
            return np.zeros(utils.shape)
        probabilities = [np.array(player.distribution.get_probabilities(), np.longdouble)
                         for player in self.chance_players]
        expected_utils = self.expected_utils()
        variance_utils = np.square(utils - expected_utils)
        for prob_arr in probabilities:
            variance_utils = np.tensordot(variance_utils, prob_arr, ([0], [0]))
        return variance_utils

    def get_utility_matrix(self):
        if len(self.chance_players) == 0:
            return self.get_utils([])
        shape = tuple([self.game.get_num_actions(player) for player in self.chance_player_indices]) + \
                tuple([self.game.get_num_actions(player) for player in self.player_indices])
        if len(self.player_indices) > 1:
            shape = shape + (len(self.player_indices),)
        utils = np.zeros(shape, np.longdouble)
        for chance_actions in itertools.product(
                *[[Action(player=player, action=action) for action in range(self.game.get_num_actions(player))]
                  for player in self.chance_player_indices]):
            index = tuple([action.action for action in chance_actions])
            utils[index] = self.get_utils(list(chance_actions))
        return utils
