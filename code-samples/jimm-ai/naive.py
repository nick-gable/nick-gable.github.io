"""
Naive agent implementation for Jimmy's Game.
Author: Nick Gable (gable105@umn.edu)

This agent operates on the following principles:
- Select the 4 highest cards for the up cards.
- Always play the highest number of our least valued card.
"""

from environment import *
NAIVE_REPORTER = True  # global variable for if we want reporting from a naive agent


class NaivePlayer(Player):
    """Naive Jimmy's Game Player."""

    def assign_hand(self, hand: Hand):
        """Take our hand, and decide face up cards."""
        self.hand = hand
        face_up_decision = np.array([], dtype="<U2")
        for card in reversed(CARD_TYPES):  # try to find highest to lowest
            remaining_spots = 4 - len(face_up_decision)
            if remaining_spots == 0:
                break

            count = np.count_nonzero(np.isin(self.hand.get_hand(), card))
            face_up_decision = np.append(face_up_decision, np.array(  # add to array
                [card] * min(count, remaining_spots)))
            remaining_spots -= min(count, remaining_spots)

        self.hand.decide_face_up(face_up_decision)

    def state_change(self, id, play_type, played_cards, pile_before, pile_after):
        if NAIVE_REPORTER and (self.id == 0):  # call default print out
            super().state_change(id, play_type, played_cards, pile_before, pile_after)

    def other_play(self, id, play_type, played_cards, pile, hands):
        """Don't need this information, but may be helpful to log it for debugging"""
        if NAIVE_REPORTER and (self.id == 0 or (self.id == 1 and id == 0)):
            # print only if we are registered first (TODO: remove this?)
            super().other_play(id, play_type, played_cards, pile, hands)

    def your_play(self, hands: "list[Hand]", pile: np.array):
        """Decide what to play, using naive strategy."""
        # decide where we need to play from
        if len(self.hand.get_hand()) > 0:
            play_type = Play.PLAYED_FROM_HAND
            card_pool = self.hand.get_hand()
        elif len(self.hand.get_face_up()) > 0:
            play_type = Play.PLAYED_FROM_UP
            card_pool = self.hand.get_face_up()
        else:  # we get to play from down cards, fun fun
            return Play.PLAYED_FROM_DOWN, np.array([])

        # okay, now try to play highest possible card from card pool
        # if we cannot, we must pick up
        for card in CARD_TYPES:
            # check if we can play this card
            # 2 guarantees we can play
            top_card = pile[-1] if len(pile) > 0 else '2'
            if np.isin(card_pool, card).any() and card_greater(card, top_card):
                # we have card, and it is greater than current pile item - play it
                played_cards = np.array(
                    [card] * np.count_nonzero(np.isin(card_pool, card)))
                return (play_type, played_cards)

        # uh oh, couldn't find a card - pick up
        return (Play.PICKED_UP, np.array([]))


if __name__ == "__main__":
    players = []
    env = Environment()

    for i in range(4):
        players.append(NaivePlayer())
        env.register_player(players[i])

    #env.register_player(Player())

    env.play_game()

    print(env.out_order)
