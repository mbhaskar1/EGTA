import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import EGTA
from DiscreteDistribution import DirichletDiscreteDistribution
from EmpiricalGame import EmpiricalGame, ChancePlayer, Noise, NoiseFactor, NoiseType
from GamutGame import GamutGame

c = 200

game = GamutGame('CovariantGame',
                 '-players', '5',
                 '-actions', '13',
                 '-r', '0.9',
                 '-random_params',
                 '-normalize',
                 '-max_payoff', f'{int(c/2)}',
                 '-min_payoff', f'{int(-c/2)}')

empirical_game = EmpiricalGame(game, [
    ChancePlayer(i, DirichletDiscreteDistribution([1] * game.get_num_actions(i))) for i in [1]
])

# empirical_game = EmpiricalGame(game, [], [
#     Noise(NoiseFactor.uniform, NoiseType.additive, scipy.stats.uniform, {'loc': -1, 'scale': 2}),
#     Noise(NoiseFactor.agent, NoiseType.additive, scipy.stats.uniform, {'loc': -1, 'scale': 2}),
#     Noise(NoiseFactor.agent_action, NoiseType.additive, scipy.stats.uniform, {'loc': -1, 'scale': 2}),
#     Noise(NoiseFactor.profile, NoiseType.additive, scipy.stats.uniform, {'loc': -0.25, 'scale': 0.5}),
#     Noise(NoiseFactor.complete, NoiseType.additive, scipy.stats.uniform, {'loc': -0.25, 'scale': 0.5})
# ])

actual = []
hoeffding = []
rademacher = []
bennett = []
talagrand = []

two_mc_era = []
two_rad = []


delta = 0.05

wimpy_variance = np.max(empirical_game.variance_utils())

# sample_sizes = list([int(100 * (1.5 ** n)) for n in range(13)])
batch_size = 500
sample_sizes = []

# expected = empirical_game.expected_utils()
streaming_object = None
for N in range(1000):
    sample_sizes.append((N+1) * batch_size)
    empirical = np.array([empirical_game.sample_utils() for _ in range(batch_size)])
    stats, streaming_object = EGTA.get_stats(empirical, streaming=True, streaming_object=streaming_object)
    # actual.append(np.max(np.abs(expected - sum(empirical)/N)))
    hoeffding.append(EGTA.hoeffding_bound(c, delta, stats))
    rademacher.append(EGTA.rademacher_bound(c, delta, stats))
    bennett.append(EGTA.bennet_union_bound(c, delta, stats))
    talagrand_bound, two_rad_val = EGTA.talagrand_bound(c, delta, stats)
    talagrand.append(talagrand_bound)

    two_mc_era.append(2 * stats['centralized_mc_era'])
    two_rad.append(two_rad_val)

    print(N)
    if (N+1) % 50 == 0:
        plt.plot(sample_sizes, hoeffding)
        plt.plot(sample_sizes, rademacher)
        plt.plot(sample_sizes, bennett)
        plt.plot(sample_sizes, talagrand)
        plt.legend(['Hoeffding', 'Rademacher', 'Bennett', 'Talagrand'])
        plt.title(f'Range: [{int(-c / 2)}, {int(c / 2)}], Wimpy Variance: {wimpy_variance}')
        plt.show()

        plt.loglog(sample_sizes, hoeffding)
        plt.loglog(sample_sizes, rademacher)
        plt.loglog(sample_sizes, bennett)
        plt.loglog(sample_sizes, talagrand)
        plt.legend(['Hoeffding', 'Rademacher', 'Bennett', 'Talagrand'])
        plt.title(f'Range: [{int(-c / 2)}, {int(c / 2)}], Wimpy Variance: {wimpy_variance}')
        plt.show()

        plt.loglog(sample_sizes, two_mc_era)
        plt.loglog(sample_sizes, two_rad)
        plt.loglog(sample_sizes, talagrand)
        plt.grid()
        plt.legend(['2 * MC_ERA', '2 * lambda/(1-2b(m))', 'Talagrand'])
        plt.show()

