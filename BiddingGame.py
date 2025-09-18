import scipy.stats

from EmpiricalGame import EmpiricalGame
import random
import heapq
from itertools import product
import numpy as np

FIRST_PRICE = 1
SECOND_PRICE = 2


class BiddingGame(EmpiricalGame):
    def __init__(self, num_players, num_goods, max_value, shading_factors, auction_type=1, random_distributions=False):
        self.num_players = num_players
        self.num_goods = num_goods
        self.max_value = max_value
        self.shading_factors = shading_factors
        self.auction_type = auction_type
        self.expected_utils = None
        self.random_distributions = random_distributions
        if self.random_distributions:
            self.alphas = []
            self.betas = []
            for p in range(self.num_players):
                self.alphas.append(random.random() * 1.5 + 0.25)
                self.betas.append(random.random() * 1.5 + 0.25)

    def get_c(self):
        return self.max_value * self.num_goods

    def get_utils_shape(self):
        actions = [len(self.shading_factors)] * self.num_players
        return tuple([*actions, self.num_players])

    def sample_homogenous_valuations(self, player=None):
        marginal_values = []
        max_value = self.max_value
        if player is None or not self.random_distributions:
            alpha = beta = 1
        else:
            alpha = self.alphas[player]
            beta = self.betas[player]
        for m in range(self.num_goods):
            marginal_values.append(scipy.stats.beta.rvs(alpha, beta) * max_value)
            max_value = marginal_values[-1]
        return marginal_values

    def sample_utils(self):
        sample_utils = np.empty(self.get_utils_shape(), dtype=np.float64)
        marginal_values = []
        for p in range(self.num_players):
            marginal_values.append(self.sample_homogenous_valuations(player=p))
        for shading_factor_indices in product(range(len(self.shading_factors)), repeat=self.num_players):
            shading_factors = [self.shading_factors[i] for i in shading_factor_indices]
            goods_won = [0] * self.num_players
            price_paid = [0] * self.num_players
            for m in range(self.num_goods):
                bids = [shading_factors[i] * marginal_values[i][goods_won[i]] for i in range(self.num_players)]
                if self.auction_type == FIRST_PRICE:
                    price = max(bids)
                    tied_players = [idx for idx in range(self.num_players) if bids[idx] == price]
                    winner = random.choice(tied_players)
                else:
                    max_bid, price = heapq.nlargest(2, bids)
                    tied_players = [idx for idx in range(self.num_players) if bids[idx] == max_bid]
                    winner = random.choice(tied_players)
                goods_won[winner] += 1
                price_paid[winner] += price
            sample_utils[shading_factor_indices] = [sum(marginal_values[i][:goods_won[i]]) - price_paid[i] for i in range(self.num_players)]
        return sample_utils

    def expected_utilities(self, m=10000):
        print(f'EXPECTED UTILS APPROXIMATED USING {m} SAMPLES')
        expected_utils = np.zeros(self.get_utils_shape(), dtype=np.float64)
        for i in range(m):
            if (i+1) % 1000 == 0:
                print(i+1)
            expected_utils = expected_utils * (i / float(i + 1)) + self.sample_utils() / (i + 1)
        self.expected_utils = expected_utils
        return expected_utils

    def variance_utils(self, m=30000):
        print(f'VARIANCE UTILS APPROXIMATED USING {m} SAMPLES')
        empirical_means = np.zeros(self.get_utils_shape(), dtype=np.float64)
        big_m_2 = np.zeros(self.get_utils_shape(), dtype=np.float64)
        for n in range(1, m + 1):
            if n % 1000 == 0:
                print(n)
            utils = self.sample_utils()
            diff = utils - empirical_means
            empirical_means += diff / n
            diff_2 = utils - empirical_means
            big_m_2 += diff * diff_2
        return big_m_2 / (m - 1)


# game = BiddingGame(3, 3, 127, [0.3, 0.5, 0.7], auction_type=FIRST_PRICE)
# e_utils = game.expected_utilities()
# print(e_utils[:, :, 0, 0])
# print(e_utils[:, :, 1, 0])
# print(e_utils[:, :, 2, 0])
