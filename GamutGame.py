import subprocess
from typing import List, Optional

import numpy as np


class Action:
    """Helper class for representing actions by players"""
    def __init__(self, player: int, action: int):
        """Initializes Action object

        :param player: Player that is taking the action
        :param action: The number corresponding to the action being taken
        """
        self.player = player
        self.action = action

    def __str__(self):
        return f"{'{'}Player: {self.player}, Action: {self.action}{'}'}"

    def __repr__(self):
        return f"{'{'}Player: {self.player}, Action: {self.action}{'}'}"


class GamutGame:
    """Class for generating and accessing GAMUT games"""

    def __init__(self, game_name, c, *args):
        """Initializes GamutGame object

        :param game_name: Name of GAMUT game to be generated
        :param args: Command Line Arguments for game to be generated
        """
        self.c = c
        if game_name == 'CustomGame':
            # If Custom Game, args should contain only utility matrix
            self.utils = args[0]
            self.players = self.utils.shape[-1]
            self.num_actions = [self.utils.shape[i] for i in range(len(self.utils.shape) - 1)]
            return

        while True:
            subprocess.call(['java', '-jar', './gamut.jar', '-g', game_name, *args, '-f', 'g.out'], shell=True)
            with open('g.out') as f:
                for line in f:
                    if line[0] == '#':
                        line = line.split()
                        if line[1] == 'Players:':
                            self.players = int(line[2])
                        elif line[1] == 'Actions:':
                            self.num_actions = list(map(int, line[2:]))
                            self.utils = np.empty((*self.num_actions, self.players))
                    else:
                        index = line.find(']')
                        action = tuple(map(lambda x: int(x) - 1, line[1:index].split()))
                        util = list(map(float, line[line.find('[', index + 1) + 1:line.find(']', index + 1)].split()))
                        self.utils[action] = util

            # If nan issue doesn't happen, we are done. Otherwise, create new game
            if not np.isnan(self.utils).any():
                break
            print('BAD GAME, CREATING NEW GAME')

    def get_c(self):
        """Get range length of utilities, c

        :return: Range length of utilities
        """
        return self.c

    def get_players(self):
        """Get the number of players

        :return: Number of players
        """
        return self.players

    def get_num_actions(self, player=None):
        """Get the number of actions available to each player, or to one particular player

        :param player: A player index (defaults to None)
        :return: Either a list containing the number of actions available to each player, or the number of actions available to the specified player
        """
        if player is not None:
            return self.num_actions[player]
        return self.num_actions

    def get_utils(self, actions: Optional[List[Action]] = None, players: Optional[List[int]] = None):
        """Get a restricted section of the utilities in the normal form representation of the game.

        :param actions: A list of actions (defaults to no restriction by action)
        :param players: A list of player indices (defaults to no restriction by player)

        :return: A numpy array containing certain utilities from the normal form representation of the game.

        If no arguments are given, all utilities are returned.

        If actions are provided, only utilities corresponding to those actions being taken are returned

        If players are provided, only utilities corresponding to those players are returned

        The beginning axes of the numpy array correspond to the actions of players for whom an action
        is not provided. These axes are ordered by the player indexes of their respective players.
        If utilities for more than one player are being returned, then there will be an additional
        axis at the end with each location corresponding to the respective requested player.
        """
        acting_players = []
        player_actions = []
        if actions is not None:
            acting_players = [action.player for action in actions]
            player_actions = [action.action for action in actions]
        slicing_indices = tuple()
        for p in range(self.players):
            if p in acting_players:
                action = player_actions[acting_players.index(p)]
                slicing_indices += (action,)
            else:
                slicing_indices += (slice(None),)
        if players is not None:
            if len(players) == 1:
                return self.utils[slicing_indices][..., players[0]]
            else:
                return self.utils[slicing_indices][..., players]
        return self.utils[slicing_indices]
