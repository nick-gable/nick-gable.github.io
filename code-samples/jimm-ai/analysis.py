"""
Module used to analyze the results used in the paper.

Author: Nick Gable (gable105@umn.edu)
"""

import json
import os
from glob import glob
from environment import Play, card_greater

RESULTS_DIR = "results"
NUM_PLAYERS = [3, 5, 9, 17]
FILE_SIZE = 100  # 100 games per file
TOTAL_PLAYS = 400


def get_results(num_players, file_size):
    files = glob(os.path.join(
        RESULTS_DIR, f"results-{num_players}p-{file_size}x-*.json"))

    file_contents = []
    for file in files:
        with open(file, 'r') as f:
            file_contents.append(json.loads(f.read()))

    return file_contents


def percent_success():
    for player_count in NUM_PLAYERS:
        success_count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            success_count += FILE_SIZE - file["jimmy_count"]

        percent = round(100 * success_count / TOTAL_PLAYS, 2)
        print(f"Percent success for {player_count} players: {percent}")


def percent_games_with_unique():
    for player_count in NUM_PLAYERS:
        count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            res = file
            for play_through in res["play_throughs"]:
                if len(play_through["differing_plays"]) > 0:
                    count += 1

        percent = round(100 * count / TOTAL_PLAYS, 2)
        print(
            f"Percent games with unique play, {player_count} players: {percent}")


def average_unique_plays():
    for player_count in NUM_PLAYERS:
        count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            res = file
            for play_through in res["play_throughs"]:
                count += len(play_through["differing_plays"])

        avg = round(count / TOTAL_PLAYS, 2)
        print(
            f"Average number of unique plays, {player_count} players: {avg}")


# up-carding: if value(expected) < value(actual)
# avoidable pick up: any time actual play is a pick up
# conservative card stacking: len(actual) < count(card in players hand)
def up_carding():
    for player_count in NUM_PLAYERS:
        use_count = 0
        win_count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            res = file
            for play_through in res["play_throughs"]:
                for play in play_through["differing_plays"]:
                    if (
                        (Play(play[1][0]) == Play.PLAYED_FROM_HAND or Play(play[1][0]) == Play.PLAYED_FROM_UP) and
                        # the expected card is not >= the actual card, so it is less
                        (not card_greater(play[0][1][0], play[1][1][0]))
                    ):
                        # do this then break to get to next play through
                        use_count += 1
                        if play_through["jimmy"] == False:
                            win_count += 1
                        break

        avg = round(win_count * 100 / use_count, 2)
        print(
            f"UC: {player_count}: % used: {round(100*use_count/TOTAL_PLAYS, 2)}, % win: {avg}")


def avoidable_pick_up():
    for player_count in NUM_PLAYERS:
        use_count = 0
        win_count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            res = file
            for play_through in res["play_throughs"]:
                for play in play_through["differing_plays"]:
                    if Play(play[1][0]) == Play.PICKED_UP and Play(play[0][0]) != Play.PICKED_UP:
                        # do this then break to get to next play through
                        print(f"Play: {play}")
                        use_count += 1
                        if play_through["jimmy"] == False:
                            win_count += 1
                        break

        avg = round(win_count * 100 / use_count, 2)
        print(
            f"AP: {player_count}: % used: {round(100*use_count/TOTAL_PLAYS, 2)}, % win: {avg}")


def conservative_card_stacking():
    for player_count in NUM_PLAYERS:
        use_count = 0
        win_count = 0
        res = get_results(player_count, FILE_SIZE)
        for file in res:
            res = file
            for play_through in res["play_throughs"]:
                for play in play_through["differing_plays"]:
                    if (
                        (Play(play[1][0]) == Play.PLAYED_FROM_HAND or Play(play[1][0]) == Play.PLAYED_FROM_UP) and
                        # compare length of actual play to count of that card in hand
                        (len(play[1][1]) < play[2]["our_hand"].count(play[1][1][0]))
                    ):
                        # do this then break to get to next play through
                        use_count += 1
                        if play_through["jimmy"] == False:
                            win_count += 1
                        break

        avg = round(win_count * 100 / use_count, 2)
        print(
            f"CCS: {player_count}: % used: {round(100*use_count/TOTAL_PLAYS, 2)}, % win: {avg}")

