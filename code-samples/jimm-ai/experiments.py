"""
Module used for running preliminary experiments.
Author: Nick Gable (gable105@umn.edu)
"""

from environment import *
import advanced
import naive

from advanced import *
from naive import *
from tqdm import tqdm

NAIVE_ONLY_COUNT = 250
ADVANCED_TEST_1_COUNT = 150

advanced.ADV_LOGGING = False
naive.NAIVE_REPORTER = False

# run simulations on just the naive agents to get a baseline
counts = [0, 0, 0]
for _ in tqdm(range(NAIVE_ONLY_COUNT), desc="Running naive-only simulations"):
    env = Environment()
    for _ in range(3):
        env.register_player(NaivePlayer())
    env.play_game()

    jimmy = env.out_order[-1]
    counts[jimmy] += 1

print("Results of naive-only test: ")
for (i, count) in enumerate(counts):
    print(
        f"Player {i}: {count} / {sum(counts)} ({(count / sum(counts)) * 100}%)")

# run simulations with advanced agent playing last
counts = [0, 0, 0]
for _ in tqdm(range(ADVANCED_TEST_1_COUNT), desc="Running simple test with MCTS agent as player 2"):
    env = Environment()
    for i in range(3):
        agent = AdvancedPlayer(3) if i == 2 else NaivePlayer()
        env.register_player(agent)
    try:
        env.play_game()
    except:
        continue

    jimmy = env.out_order[-1]
    counts[jimmy] += 1

print("Results of first MCTS test (player 2 = MCTS): ")
for (i, count) in enumerate(counts):
    print(
        f"Player {i}: {count} / {sum(counts)} ({(count / sum(counts)) * 100}%)")
