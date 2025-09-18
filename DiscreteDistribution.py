from typing import List, Union

import numpy as np


class DiscreteDistribution:
    """Parent class for discrete distributions. Can be used to create a directly defined discrete distribution."""

    def __init__(self, probabilities: List[float]):
        """Initializes DiscreteDistribution object forming a discrete distribution over 0, 1, ..., N-1

        :param probabilities: List of probabilities for elements 0, 1, ..., N-1
        """
        self.N = len(probabilities)
        self.probabilities = probabilities

    def get_probability(self, x) -> float:
        """Get probability of x being sampled.

        :param x: Number within domain of discrete distribution
        :return: Probability of x being sampled
        """
        return self.probabilities[x]

    def get_probabilities(self) -> List[float]:
        """Get probabilities of each respective element in the domain of discrete distribution being sampled.

        :return: List of probabilities corresponding to each element in domain of discrete distribution
        """
        return self.probabilities

    def sample(self, n=1) -> Union[int, List[int]]:
        """Generate n samples from discrete distribution

        :param n: Number of samples to generate (default 1)
        :return: Numpy array containing integer samples, or single integer sample
        """
        res = np.random.choice(list(range(self.N)), n, p=self.probabilities)
        if n == 1:
            return res[0]
        else:
            return res


class DirichletDiscreteDistribution(DiscreteDistribution):
    """Child of DiscreteDistribution that generates random Discrete Distributions using the Dirichlet Distribution."""

    def __init__(self, alphas: List[float]):
        """Generates a random Discrete Distribution using the Dirichlet Distribution.

        :param alphas: List of positive concentration parameters for Dirichlet Distribution
        """
        super().__init__(np.random.dirichlet(alphas))

