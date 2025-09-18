from unittest import TestCase

import numpy as np
import scipy.stats

from EGTA import hoeffding_bound, rademacher_bound, mc_era, centralized_mc_era, get_stats
from EmpiricalGame import Noise, NoiseFactor, NoiseType
from Algorithms import *
from Games import construct_empirical_game, congestion_game, congestion_facilities_to_action

sample_utils = [
    np.array([[1, 2, 3], [-1, 0, 1], [2, -1, 0]]),
    np.array([[2, 1, 2], [0, 1, -1], [3, 0, -2]]),
    np.array([[0, 3, -1], [1, 2, 1], [-1, 0, 2]]),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    np.array([[-1, -2, -3], [2, 0, 3], [0, -1, 2]])
]
c = 20
d = 0.01


class EGTA(TestCase):
    def test_hoeffding(self):
        stats = get_stats(np.array(sample_utils[:3]))
        self.assertAlmostEqual(hoeffding_bound(c, d, stats), 22.3540331, places=5)

    def test_rademacher(self):
        stats = get_stats(np.array(sample_utils[:3]), seed=5)
        # with this seed the three random rademacher variables are -1, 1, -1
        self.assertAlmostEqual(rademacher_bound(c, d, stats), 55.2318844, places=5)

    def test_mc_era(self):
        np.random.seed(1)
        sample_utils_np = np.array(sample_utils[:3])
        self.assertAlmostEqual(mc_era(2, sample_utils_np), 5 / 3, places=5)

    def test_mc_era_streaming(self):
        np.random.seed(1)
        _, streaming_object = mc_era(2, np.array([sample_utils[0], sample_utils[1]]), streaming=True)
        result, _ = mc_era(2, np.array([sample_utils[2]]), streaming=True, streaming_object=streaming_object,
                           return_result=True)
        self.assertAlmostEqual(result, 5 / 3, places=5)

    def test_centralized_mc_era_zero_case(self):
        sample_utils_np = np.array([sample_utils[0] for i in range(10)])
        sample_mean = np.mean(sample_utils_np, axis=0)
        self.assertAlmostEqual(centralized_mc_era(5, sample_utils_np, sample_mean), 0.0)

    def test_get_stats(self):
        seed = 3
        stats = get_stats(np.array(sample_utils[:5]), mc_era_n=2, seed=seed)
        self.assertEqual(stats['m'], 5)
        print(-sum(sample_utils[:5])/25)
        np.testing.assert_array_almost_equal_nulp(stats['sample_mean'], sum(sample_utils[:5]) / 5, nulp=5)
        np.testing.assert_array_almost_equal_nulp(stats['sample_variance'], np.array([
            [1.3, 3.5, 5.8],
            [1.3, 0.7, 2],
            [2.5, 0.7, 2.8]
        ]))
        self.assertAlmostEqual(stats['empirical_wimpy'], 4.64, places=5)
        self.assertAlmostEqual(stats['centralized_maximum'], 3.4, places=5)
        self.assertAlmostEqual(stats['one_era'], 1.4, places=5)
        self.assertAlmostEqual(stats['centralized_mc_era'], 1.2, places=5)
        self.assertEqual(stats['cardinality'], 9)

    def test_get_stats_streaming(self):
        seed = 3
        _, streaming_object = get_stats(np.array(sample_utils[:2]), streaming=True, return_results=False, seed=seed,
                                        mc_era_n=2)
        stats, _ = get_stats(np.array(sample_utils[2:5]), streaming=True, streaming_object=streaming_object,
                             return_results=True, seed=seed + 1, mc_era_n=2)
        self.assertEqual(stats['m'], 5)
        np.testing.assert_array_almost_equal_nulp(stats['sample_mean'], sum(sample_utils[:5]) / 5, nulp=5)
        np.testing.assert_array_almost_equal_nulp(stats['sample_variance'], np.array([
            [1.3, 3.5, 5.8],
            [1.3, 0.7, 2],
            [2.5, 0.7, 2.8]
        ]))
        self.assertAlmostEqual(stats['empirical_wimpy'], 4.64, places=5)
        self.assertAlmostEqual(stats['centralized_maximum'], 3.4, places=5)
        self.assertAlmostEqual(stats['one_era'], 1, places=5)
        self.assertAlmostEqual(stats['centralized_mc_era'], 1, places=5)
        self.assertEqual(stats['cardinality'], 9)

    def test_uniform_noise(self):
        arr = np.ones((3, 3, 2))
        noise = Noise(NoiseFactor.uniform, NoiseType.additive, scipy.stats.norm, {'loc': 0.4, 'scale': 0.00000001})
        noise.apply_noise(arr)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    self.assertAlmostEqual(arr[i, j, k], 1.4, places=3)

    def test_agent_noise(self):
        # with random_state = 1, norm.rvs produces [1.62434536 -0.61175641]
        arr = np.ones((3, 3, 2))
        noise = Noise(NoiseFactor.agent, NoiseType.additive, scipy.stats.norm, {'random_state': 1})
        noise.apply_noise(arr)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(arr[i, j, 0], 1 + 1.62434536, places=5)
                self.assertAlmostEqual(arr[i, j, 1], 1 - 0.61175641, places=5)

    def test_agent_action_noise(self):
        # with random_state = 1, loc=1, scale=0.5, norm.rvs produces [1.81217268 0.69412179 0.73591412]
        arr = np.ones((3, 3, 2))
        noise = Noise(NoiseFactor.agent_action, NoiseType.multiplicative, scipy.stats.norm,
                      {'loc': 1, 'scale': 0.5, 'random_state': 1}, agent_action_selector=1)
        noise.apply_noise(arr)
        for i in range(3):
            for k in range(2):
                self.assertAlmostEqual(arr[i, 0, k], 1.81217268, places=5)
                self.assertAlmostEqual(arr[i, 1, k], 0.69412179, places=5)
                self.assertAlmostEqual(arr[i, 2, k], 0.73591412, places=5)

    def test_profile_noise(self):
        # with random_state = 1, norm.rvs produces:
        rvs = np.array([[1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862, 0.86540763, -2.3015387],
                        [1.74481176, -0.7612069, 0.3190391]])
        arr = np.ones((3, 3, 2))
        noise = Noise(NoiseFactor.profile, NoiseType.additive, scipy.stats.norm, {'random_state': 1})
        noise.apply_noise(arr)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    self.assertAlmostEqual(arr[i, j, k], 1 + rvs[i, j], places=5)

    def test_complete_noise(self):
        rvs = scipy.stats.norm.rvs(size=(3, 3, 2), random_state=1)
        arr = np.ones((3, 3, 2))
        noise = Noise(NoiseFactor.complete, NoiseType.multiplicative, scipy.stats.norm, {'random_state': 1})
        noise.apply_noise(arr)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    self.assertAlmostEqual(arr[i, j, k], rvs[i, j, k], places=5)

    def test_regret(self):
        utils = np.moveaxis(np.array(sample_utils[:2]), 0, 2)
        np.testing.assert_array_almost_equal_nulp(regret(utils, [None, 1], 0), np.array([0, 2, 3]), nulp=5)
        np.testing.assert_array_almost_equal_nulp(regret(utils, [2, None], 1, [1]), np.array([3]), nulp=5)

    def test_regrets_matrix(self):
        utils = np.moveaxis(np.array(sample_utils[:2]), 0, 2)
        true_regrets = np.array([
            [[1, 0], [0, 1], [0, 0]],
            [[3, 1], [2, 0], [2, 2]],
            [[0, 0], [3, 3], [3, 5]]
        ])
        algo_regrets = regrets_matrix(utils)
        np.testing.assert_array_almost_equal_nulp(algo_regrets, true_regrets, nulp=5)

    def test_epsilon_sensitive_regrets_matrix(self):
        utils = np.array([
            [[1, 1], [0, 3]],
            [[4, 0], [2, 2]]
        ])
        epsilon = np.array([
            [[0.5, 0.2], [0.3, 0.4]],
            [[0.1, 1], [0.9, 1]]
        ])
        true_epsilon_regrets = np.array([
            [[2.4, 1.4], [0.8, 0]],
            [[0, 0], [0, 0]]
        ])
        algo_epsilon_regrets = regrets_lower_bound_matrix(utils, epsilon)
        np.testing.assert_array_almost_equal_nulp(algo_epsilon_regrets, true_epsilon_regrets, nulp=5)

    def test_get_c(self):
        game = construct_empirical_game(congestion_game(2, 3, c=200), uniform_noise=100, agent_noise=50, complete_noise=100)
        self.assertEqual(game.get_c(), 700)

    def test_eps_nash_equilibria(self):
        utils = np.array([
            [[1, 1], [3, 0]],
            [[0, 4], [2.6, 3.5]]
        ])
        eps_nash = eps_nash_equilibria(utils, 0.5)
        true_eps_nash = np.array([
            [1, 0],
            [0, 1]
        ])
        np.testing.assert_array_equal(eps_nash, true_eps_nash)

    def test_congestion_facilities_to_action(self):
        self.assertEqual(congestion_facilities_to_action([0]), 0)
        self.assertEqual(congestion_facilities_to_action([1]), 1)
        self.assertEqual(congestion_facilities_to_action([3, 1, 4]), 25)

    def test_maximum_adjacent_var(self):
        variances = np.array([
            [[1, 1], [2, 3], [0, 1]],
            [[2, 1], [1, 2], [1, 0]],
            [[2, 2], [1, 2], [3, 3]]
        ])
        self.assertListEqual(maximum_adjacent_var(variances, [None, 0], 0).tolist(), [2, 2, 2])
        self.assertListEqual(maximum_adjacent_var(variances, [2, None], 1).tolist(), [3, 3, 2])

    def test_maximum_adjacent_var_player_matrix(self):
        variances = np.array([
            [[1, 1], [2, 3], [0, 1]],
            [[2, 1], [1, 2], [1, 0]],
            [[2, 2], [1, 2], [3, 3]]
        ])
        np.testing.assert_array_equal(
            maximum_adjacent_var_player_matrix(variances, 0),
            np.array([
                [2, 1, 3],
                [2, 2, 3],
                [2, 2, 1]
            ])
        )
        np.testing.assert_array_equal(
            maximum_adjacent_var_player_matrix(variances, 1),
            np.array([
                [3, 1, 3],
                [2, 1, 2],
                [3, 3, 2]
            ])
        )
