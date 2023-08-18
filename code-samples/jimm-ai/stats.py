"""
Statistics taking utilities for Jimmy's Game. So far, used
to track average hand size during games.
Author: Nick Gable (gable105@umn.edu)
"""

from environment import *
from naive import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class StatEnvironment(Environment):
    """Environment that overrides advance_play() to collect hand size
    statistics."""

    def __init__(self):
        self.avg_size = np.array([])
        self.size_at_endgame = None
        super().__init__()

    def advance_play(self):
        if not self.size_at_endgame and len(self.players_remaining) == 2:
            self.size_at_endgame = sum([len(self.hands[i].hand) + len(self.hands[i].face_down) + len(self.hands[i].face_up)
                                        for i in self.players_remaining]) / len(self.players_remaining)
        
        avg = sum([len(self.hands[i].hand) + len(self.hands[i].face_down) + len(self.hands[i].face_up)
                  for i in self.players_remaining]) / len(self.players_remaining)
        self.avg_size = np.append(self.avg_size, avg)
        super().advance_play()

    def plot(self):
        """Plot the average hand size as a function of time in the game."""
        plt.plot(np.arange(0, len(self.avg_size)), self.avg_size)
        plt.xlabel("Turn number")
        plt.ylabel("Average hand size")
        plt.title("Average hand size as game progresses")
        plt.show()


if __name__ == "__main__":
    vals = np.array([])
    vals_2 = np.array([])
    for _ in tqdm(range(500), desc="Running game simulations"):
        env = StatEnvironment()
        for i in range(5):
            env.register_player(NaivePlayer())

        env.play_game()

        vals = np.append(vals, env.size_at_endgame)
        vals_2 = np.append(vals_2, env.avg_size[-10:].mean())
    
    plt.boxplot(vals)
    plt.show()

    plt.boxplot(vals_2)
    plt.show()
