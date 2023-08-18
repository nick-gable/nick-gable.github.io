"""
Module used for the final testing experiments, directly used by the paper.
Author: Nick Gable (gable105@umn.edu)
"""
from environment import *
import advanced
import naive
import json
import os
import timeit

from advanced import *
from naive import *
from multiprocessing import Process
from random import random

# logging controls
advanced.ADV_LOGGING = False
naive.NAIVE_REPORTER = False
TEST_DEBUG = False

# Testing parameters
# play a small, medium, large, and very large game
NUM_PLAYERS = [3, 5, 9, 17]

# number of iterations per game size
NUM_ITERATIONS = 200

RESULTS_DIR = 'results'


def log(name: str, msg: str):
    print(f"[{name}] {msg}")


class TestingEnvironment(Environment):
    """Environment tuned for our paper's experimental setup."""

    def __init__(self):
        """Initialize special data structures."""

        self.activated_agent = False  # set to True once we have activated one

        # list of (expected, actual, state) tuples for when advanced agent plays something unique.
        self.differing_plays = []

        super().__init__()

    def expected_outcome(self, player_id: int):
        """Returns the expected play that the naive agent would make for a given player ID given the current game state."""
        test_player = NaivePlayer()
        test_player.hand = self.hands[player_id]
        test_player.assign_id(player_id)
        return test_player.your_play(self.hands, self.pile)

    def get_state(self):
        """Generates a concise output of current game state - only call this if two players are remaining!"""
        opp_up_cards = list(self.hands[self.naive_idx].get_face_up())
        opp_hand_up = list(self.hands[self.naive_idx].get_hand())
        opp_down_count = self.hands[self.naive_idx].number_down()

        our_hand_up = list(self.hands[self.adv_idx].get_hand())
        our_up_cards = list(self.hands[self.adv_idx].get_face_up())
        our_down_count = self.hands[self.adv_idx].number_down()

        down_cards = list(self.players[self.adv_idx].cards_remaining)
        pile = list(self.pile)

        return {'our_hand': our_hand_up, 'our_up_cards': our_up_cards, 'our_down_count': our_down_count, 'opp_up_cards': opp_up_cards,
                'opp_hand': opp_hand_up, 'opp_down_count': opp_down_count, 'down_cards': down_cards, 'pile': pile}

    def advance_play(self):
        """Main difference here: If there are only two players left, 
        deactivate one of them and record the other one as the advanced AI player."""
        expected_play = None

        if len(self.players_remaining) == 2 and not self.activated_agent:
            # randomly select an agent to deactivate
            players_rem_idx = 0 if random(
            ) < 0.5 else 1
            players_rem_adv_idx = 1 if players_rem_idx == 0 else 0

            naive_idx = self.players_remaining[players_rem_idx]
            adv_idx = self.players_remaining[players_rem_adv_idx]

            self.players[naive_idx].deactivate()
            self.adv_idx = adv_idx
            self.naive_idx = naive_idx

            self.activated_agent = True

            if TEST_DEBUG:
                print(f"Selecting player {adv_idx} to be advanced AI")

        if len(self.players_remaining) == 2 and self.turn == self.adv_idx:
            expected_play = self.expected_outcome(self.adv_idx)
            state = self.get_state()  # record state before play, its more useful

        # advance play so that last play is up to date
        super().advance_play()

        if expected_play is not None:
            difference = False
            (exp_pt, exp_cards) = expected_play
            (act_pt, act_cards) = self.last_play

            exp_cards.sort()
            act_cards.sort()

            e_str = '-'.join(exp_cards)
            a_str = '-'.join(act_cards)

            if exp_pt != act_pt or e_str != a_str:
                difference = True

            if difference:
                # play difference, record it
                expected_play = (exp_pt, list(exp_cards))
                self.last_play = (act_pt, list(act_cards))
                self.differing_plays.append(
                    (expected_play, self.last_play, state))
                if TEST_DEBUG:
                    print(
                        f"Play recorded as different: {self.differing_plays[-1]}")


def run_test(num_players: int, num_iters: int):
    """Run a test with specified number of players, specified number of times"""
    def t_log(msg): return log(f"test {num_players}p/{num_iters}x", msg)

    t_log("begin test")

    # number of times advanced AI is jimmy (50% is baseline)
    advanced_jimmy = 0
    results = {'total_plays': num_iters, 'play_throughs': []}
    try:
        for i in range(num_iters):
            now = timeit.default_timer()
            env = TestingEnvironment()
            for _ in range(num_players):
                env.register_player(AdvancedPlayer(num_players))

            env.play_game()
            if TEST_DEBUG:
                t_log(f"end of round, out order is {env.out_order}")
            if env.out_order[-1] == env.adv_idx:
                """Our agent was Jimmy, record that"""
                advanced_jimmy += 1

            results["play_throughs"].append({"jimmy": (
                env.out_order[-1] == env.adv_idx), 'differing_plays': env.differing_plays})

            t_log(
                f"done with simulation {i}, {round(timeit.default_timer() - now, 2)}s")
    except Exception as err:
        t_log(f"Error, exiting and printing at iter {i}: {str(err)}")

    results["jimmy_count"] = advanced_jimmy

    filename = os.path.join(
        RESULTS_DIR, f"results-{num_players}p-{num_iters}x-{int(random() * 10000000)}.json")
    with open(filename, 'w') as file:
        file.write(json.dumps(results))

    t_log(f"end test, results saved to {filename}")
    return results


def tests():
    # run eight processes, each of which split half of one testing load
    processes = []
    for player_count in NUM_PLAYERS*2:
        count_per_process = int(NUM_ITERATIONS / 2)
        processes.append(Process(target=run_test, args=(
            player_count, count_per_process)))

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    tests()
