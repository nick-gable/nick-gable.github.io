"""
Advanced AI Jimmy's Game player implementation, using Monte-Carlo Tree Search.

Makes use of jimmymcts Rust library that is in the venv for this directory. Does not start playing
until there are two players left.
"""

from jimmymcts import search_best_play
from environment import *
from naive import *
ADV_LOGGING = True  # log card counting or not

class ImplementationError(Exception):
    """Uh, oh, our AI has done something bad!"""


def remove_cards(remove_from: np.ndarray, cards: np.ndarray):
    """Returns remove_from without cards."""
    for card in cards:  # iterate over each card so that they are removed one by one
        if not np.isin(card, remove_from).any():
            raise ImplementationError(
                f"Attempted to remove {cards} from {remove_from}")
        first_pos = np.where(remove_from == card)[0][0]
        remove_from = np.append(
            remove_from[0:first_pos], remove_from[first_pos+1:])
    return remove_from


class AdvancedPlayer(NaivePlayer):
    """MCTS enabled Jimmy's Game Player.

    Extends NaivePlayer because the early game play is the same."""

    def __init__(self, player_count, deck_count=None):
        """Initialize advanced player data structures on player and deck count.

        If deck_count is None, compute correct deck number"""
        if deck_count is None:
            deck_count = 1 + ((12 * player_count) // 52)

        # array used to track remaining cards in deck
        # once two players are left, this becomes the down card potential deck
        self.cards_remaining = np.array(CARD_TYPES * 4 * deck_count)

        # data structure for keeping track of cards in players hands
        # we place an 'X' if we don't know the card, and replace them with real cards we
        # (eventually) will get to know
        # of course, we will know our cards, so we don't track those here
        # Note: Currently not using 'X's - may not be needed :)
        self.card_counts = [np.array([]) for _ in range(player_count)]

        self.first_turn = True  # used for initialization on first move

        self.activated = True  # changed to False if an environment wants to prevent us from activating

    def deactivate(self):
        """Prevent this agent from using the MCTS advanced AI to play the game."""
        self.activated = False

    def first_play(self, hands: "list[Hand]"):
        """Called on first play - removes known face up cards from cards remaining."""
        if ADV_LOGGING:
            print(f"First play: All hands are {hands}")
        for (i, hand) in enumerate(hands):
            if i == self.id:
                continue  # don't count our cards, already been done!
            self.cards_remaining = remove_cards(
                self.cards_remaining, hand.get_face_up())
            if ADV_LOGGING:
                print(f"Removing face up cards for player {i}: {hand}")
        self.first_turn = False

    def assign_hand(self, hand: Hand):
        """Remove our initial cards from the cards remaining pool."""
        self.cards_remaining = remove_cards(
            self.cards_remaining, hand.get_hand())
        if ADV_LOGGING:
            print(f"Removing our cards: {hand.get_hand()}")
        super().assign_hand(hand)

    def take_cards(self, cards: np.ndarray):
        """Notifies us that we have been given cards, and removes them from counter."""
        self.cards_remaining = remove_cards(self.cards_remaining, cards)
        if ADV_LOGGING:
            print(f"Removing cards we were dealt: {cards}, hand is: {self.hand.get_hand()}")

    def state_change(self, id, play_type, played_cards, pile_before, pile_after):
        """Notify of a change in game state caused by any player's actions."""
        # in case we are logging, call this to report
        super().state_change(id, play_type, played_cards, pile_before, pile_after)
        
        if play_type == Play.PICKED_UP:
            # player picked up, add to their count
            self.card_counts[id] = np.append(self.card_counts[id], pile_before)
            if ADV_LOGGING:
                print(f"Adding {pile_before} to {id} count")
        elif play_type == Play.PLAYED_FROM_HAND:
            # if we played, just return
            if id == self.id:
                return

            # see if we knew the player had this card - if not, remove it from
            # cards remaining
            for card in played_cards:
                if np.isin(card, self.card_counts[id]).any():
                    # we know this card exists - remove it from our tracker
                    self.card_counts[id] = remove_cards(
                        self.card_counts[id], np.array([card]))
                    if ADV_LOGGING:
                        print(f"Removing known card of {card} from {id} count")
                else:
                    # hmm, new card! remove it from cards remaining
                    self.cards_remaining = remove_cards(
                        self.cards_remaining, np.array([card]))
                    if ADV_LOGGING:
                        print(f"Removing new card {card} from cards remaining")
        elif play_type == Play.PLAYED_FROM_DOWN:
            # remove card played from cards remaining
            self.cards_remaining = remove_cards(
                self.cards_remaining, played_cards)
            if ADV_LOGGING:
                print(f"Removing revealed down cards {played_cards} from cards remaining")

            # only execute next steps for other players
            if id == self.id:
                return

            if len(pile_after) == 0:
                # if so, player had to pick up - add to their tracker
                self.card_counts[id] = np.append(
                    self.card_counts[id], pile_before)
                if ADV_LOGGING:
                    print(f"Adding {pile_before} to {id} count")

                # also add down card to the player's hand, since environment gives us old pile before
                # any play
                self.card_counts[id] = np.append(self.card_counts[id], played_cards)
                if ADV_LOGGING:
                    print(f"Adding played down cards {played_cards} back to {id} count")

    def other_play(self, id, play_type, played_cards, pile, hands):
        if self.first_turn:
            self.first_play(hands)
            # self.previous_pile = pile  # we need to track this to know what gets picked up

        # adjust our internal structures based off of what was played
        if play_type == Play.PICKED_UP:
            # add all cards from the pile to our hand tracker for this player
            self.card_counts[id] = np.append(
                self.card_counts[id], self.previous_pile)
        elif play_type == Play.PLAYED_FROM_HAND:
            # played from hand: remove cards from hand if we knew it was there
            # also, remove from cards remaining in deck
            # remove a card if we haven't seen it in their hand yet - untracked up to this point
            for card in played_cards:
                if np.isin(card, self.card_counts[id]).any():
                    # we know this card exists - remove it from our tracker
                    self.card_counts[id] = remove_cards(
                        self.card_counts[id], np.array([card]))
                else:
                    # hmm, new card! remove it from cards remaining
                    self.cards_remaining = remove_cards(
                        self.cards_remaining, np.array([card]))
        elif play_type == Play.PLAYED_FROM_DOWN:
            # always reveals another card - remove it from cards remaining
            self.cards_remaining = remove_cards(
                self.cards_remaining, played_cards)

            # check if player had to pick up the pile
            if len(self.previous_pile) > 0 and not card_greater(played_cards[0], self.previous_pile[-1]):
                # if this is the case, they need to have picked up, so we add to their card count previous pile
                self.card_counts[id] = np.append(
                    self.card_counts[id], self.previous_pile)

                # also add card they played to their card count
                self.card_counts[id] = np.append(
                    self.card_counts[id], played_cards
                )

        # adjust this before we advance
        self.previous_pile = pile

    def your_play(self, hands: "list[Hand]", pile: np.array):
        """Play the naive strategy until there are only two players left, after which we use
        MCTS to (hopefully) perform better than naive strategy."""
        if self.first_turn:
            self.first_play(hands)
        
        players_remaining = 0
        remaining_id = None
        for (i, hand) in enumerate(hands):
            if not hand.is_out():
                players_remaining += 1
                if i != self.id:
                    remaining_id = i  # if only one, setting in loop is fine
            

        if players_remaining > 2 or not self.activated:
            return super().your_play(hands, pile)

        # engage MCTS, since we are down to one other player
        opp_up_cards = list(hands[remaining_id].get_face_up())
        opp_hand_up = list(self.card_counts[remaining_id])
        opp_down_count = hands[remaining_id].number_down()

        our_hand_up = list(self.hand.get_hand())
        our_up_cards = list(self.hand.get_face_up())
        our_down_count = self.hand.number_down()

        down_cards = list(self.cards_remaining)
        pile = list(pile)
        our_turn = True
        times = 10000
        debug = False

        try:
            (play_type, played_cards) = search_best_play(
                our_hand_up, our_up_cards, our_down_count, opp_hand_up, opp_up_cards,
                opp_down_count, down_cards, pile, our_turn, times, debug
            )
        except:
            print("\n---MCTS failure---")
            print(f"Cards remaining: {self.cards_remaining}")
            print(f"Parameters: ")
            parameters_list = [our_hand_up, our_up_cards, our_down_count, opp_hand_up, opp_up_cards,
                opp_down_count, down_cards, pile, our_turn, times, debug]
            parameters_names = ["our_hand_up", "our_up_cards", "our_down_count", "opp_hand_up", "opp_up_cards",
                "opp_down_count", "down_cards", "pile", "our_turn", "times", "debug"]
            print('\n'.join([f"{parameters_names[i]}: {parameters_list[i]}" for i in range(len(parameters_list))]))

        played_cards = np.array(played_cards)
        return (Play(play_type), played_cards)


if __name__ == "__main__":
    env = Environment()

    for i in range(1):
        env.register_player(NaivePlayer())

    # register three AI players so some survive to endgame
    for i in range(1):
        env.register_player(AdvancedPlayer(7))

    env.register_player(Player())

    env.play_game()

    print(env.out_order)
