from abc import ABC, abstractmethod
from typing import List


class EmpiricalGame(ABC):
    @abstractmethod
    def get_c(self):
        """Get range length of empirical utilities (affected by noise)

        :return: Range length of empirical utilities
        """
        pass

    @abstractmethod
    def get_utils_shape(self):
        """Gets shape of utility matrix for active players

        :return: Tuple of ints corresponding to shape of utility matrix
        """
        pass

    @abstractmethod
    def sample_utils(self):
        """Sample chance actions for each chance player and get resulting utilities for each combination of player
        actions

        :return: Numpy array containing utilities corresponding to each combination of player actions and each player. Look at get_utils in GamutGame for more info.
        """
        pass

    @abstractmethod
    def expected_utilities(self):
        """Get expectation over utilities with respect to the Chance Player distributions.

        :return: Numpy array containing expected utilities corresponding to each combination of player actions and each player.
        """
        pass

    @abstractmethod
    def variance_utils(self):
        pass
