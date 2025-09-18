import numpy as np
import pygambit
from Games import random_zero_sum

NUM_ACTIONS = 4
game = random_zero_sum(NUM_ACTIONS, c=2)

print(game.utils[:, :, 0])
print(game.utils[:, :, 1])

g = pygambit.Game.new_table([NUM_ACTIONS, NUM_ACTIONS])
for i in range(NUM_ACTIONS):
    for j in range(NUM_ACTIONS):
        g[i, j][0] = pygambit.Rational(game.utils[i, j, 0])
        g[i, j][1] = -g[i, j][0]
# for profile in g.contingencies:
#     print(profile, g[profile][0], g[profile][1])

sol = pygambit.nash.lp_solve(game=g, rational=False, use_strategic=True)
sol = np.array(sol).reshape(-1)

outputs = [sol[:NUM_ACTIONS], sol[NUM_ACTIONS:]]

print(outputs)

p1_values = np.einsum('ij,j->i', game.utils[..., 0], outputs[1])
p2_values = np.einsum('ij,i->j', game.utils[..., 1], outputs[0])
p1_value = np.einsum('i,i->', p1_values, outputs[0])     # 2
p2_value = np.einsum('i,i->', p2_values, outputs[1])
p1_regret = np.amax(p1_values) - p1_value
p2_regret = np.amax(p2_values) - p2_value

print(p1_regret)
print(p2_regret)

print(np.array(sol))
