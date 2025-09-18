# EGTA

This code-base provides a framework for experiments in Empirical Game-Theoretic Analaysis (EGTA), including progressive sampling-based algorithms for learning Nash equilibria and other game-theoretic properties in games with noisy access to the true utility function, which we call simulation-based games. The primary files in this code-base are as follows:

- **EGTA.py**: Provides functions for computing the essential EGTA sample-complexity bounds (e.g., Hoeffding, Bennett, Talagrand / Rademacher Average, etc.), and an efficient algorithm for updating these bounds dynamically as you collect more samples.
- **Algorithms.py**: Implementations of a variety of sampling-based algorithms (both global and progressive sampling) for learning properties in simulation-based games.
- **Experiments.py**: A wide collection of experiments we have run to try to compare and understand the performance of our various sampling algorithms.
- **good_experiments**: A list of a couple ways (parameter choices) the aforementioned experiments may be run.
- **GamutGame.py**: Class for loading and representing games through the game-generating library GAMUT.
- **Games.py**: A collection of pre-defined simple generating functions for using GAMUT to generate various kinds of games.

The code provided here has not been proof-read carefully yet, so there may be some incomplete or incorrectly defined functions. However, the core progressive sampling algorithms in **Algorithms.py** and the majority of the experiments in **Experiments.py** should be correctly implemented. This framework was used to produce the experiments in the following papers:

[1] **Learning Properties in Simulation-Based Games**		2023 <br/>
Cyrus Cousins*, Bhaskar Mishra, Enrique Areyan Viqueira Amy Greenwald <br/>
In Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS) [[PDF](https://dl.acm.org/doi/abs/10.5555/3545946.3598647)]

[2] **Regret Pruning for Learning Equilibria in Simulation-Based Games**	2022
Bhaskar Mishra*, Cyrus Cousins, Amy Greenwald <br/>
Available as arXiv preprint [[PDF](https://arxiv.org/abs/2211.16670)] <br/>

[3] **Computational and Data Requirements for Learning Generic Properties of Simulation-Based Games**	2022
Cyrus Cousins*, Bhaskar Mishra, Enrique Areyan Viqueira, Amy Greenwald <br/>
Oral Presentation at the INFORMS 2022 Sequential Auctions Workshop [[PDF](https://arxiv.org/abs/2208.06400)] <br/>
