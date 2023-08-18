from environment import *
from naive import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

x = range(2, 101)
y = []
xg = []

class CountingEnvironment(Environment):
    def __init__(self):
        self.plays = 0
        super().__init__()

    def advance_play(self):
        self.plays += 1
        return super().advance_play()

for i in tqdm(x, desc="Running game simulations"):
    for _ in range(5):
        env = CountingEnvironment()
        for _ in range(i):
            env.register_player(NaivePlayer())
        env.play_game()
        xg.append(i)
        y.append(env.plays)

plt.plot(xg, y)
plt.show()
