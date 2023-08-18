"""
Abstract environment for playing Jimmy's Game!
Author: Nick Gable (gable105@umn.edu)

See project write-up for full description of game rules.

Specific implementation notes here:
- Players start with 4 down cards.
- 2 resets, and 10 is a bomb. No 7's or 3's.
"""
import numpy as np
from enum import IntEnum

# ordered by general card value
CARD_TYPES = ['3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K', 'A', '2', '10']


class InvalidPlay(Exception):
    """Exception raised when a player attempts to make a move that is illegal. Should not occur
    in normal play, especially not under agent play, which would indicate an implementation error. Thus, this
    exception should terminate game play"""
    pass


class Deck:
    """Deck simulation class. Allows for a variable number of 52 card decks to be put into play.
    String representation of cards, no suits (not needed for Jimmy's Game)."""

    def __init__(self, num_decks):
        """Creates a new Deck with num_decks regular decks in play, and randomly creates a card ordering."""
        self.cards = np.array((CARD_TYPES * (4 * num_decks)))
        np.random.shuffle(self.cards)

    def draw(self, num_cards):
        """Draw num_cards cards, returning them as an array. If the deck runs out during the draw, only return
        the number of possible cards."""
        drawn = self.cards[0:num_cards]
        self.cards = self.cards[num_cards:]
        return drawn

    def left(self):
        """Return number of cards left to be drawn"""
        return len(self.cards)


class Hand:
    """Container used to store hand information for a Player. Methods exist to get the private cards in the hand
    (only available in normal play to the player), as well as face up cards (anyone can see), and face down cards
    (normally only known when flipped). When initialized, face up cards are empty, so that the player can decide them."""

    def __init__(self, face_down, hand):
        """Initialize the hand. face_down should be 4 cards, hand 8 (player decides which 4 to put face up ultimately)"""
        self.face_down = face_down
        self.face_up = None  # decided later
        self.hand = hand

    def __str__(self):
        """Print desired visible hand information"""
        return f"Hand with {self.number_down()} down, {self.face_up} up, {len(self.hand)} in hand"

    def __repr__(self):
        return str(self)

    def play_face_up(self, cards):
        """Removes the specified cards from the face up hand. Raises InvalidPlay if not in face up."""
        for card in cards:  # iterate over each card so that they are removed one by one
            if card not in self.face_up:
                raise InvalidPlay(
                    f"Attempted to play up card {card} that is not in up cards {self.face_up}")
            # find first position this card occurs, and remove it
            first_pos = np.where(self.face_up == card)[0][0]
            self.face_up = np.delete(self.face_up, np.array([first_pos]))
        return cards

    def play_face_down(self):
        """Removes and returns one face down card."""
        down_card = self.face_down[0]
        self.face_down = self.face_down[1:]
        return down_card

    def play_from_hand(self, cards):
        """Play specified cards from hand. Raises InvalidPlay if specified cards do not exist."""
        for card in cards:  # iterate over each card so that they are removed one by one
            if card not in self.hand:
                raise InvalidPlay(
                    f"Attempted to card {card} that is not in hand of {self.hand}")
            # find first position this card occurs, and remove it
            first_pos = np.where(self.hand == card)[0][0]
            self.hand = np.delete(self.hand, np.array([first_pos]))
        return cards

    def decide_face_up(self, face_up):
        """Specify which cards the player wants to have in face up. Raises InvalidPlay if specified cards do not exist."""
        if self.face_up is not None:
            raise InvalidPlay("Player attempted to change face up cards")

        # "plays" these cards so they are removed from hand
        self.play_from_hand(face_up)
        self.face_up = face_up

    def get_hand(self):
        """Returns the private hand for this player. Should only be accessed by the owner of this hand in normal play."""
        return self.hand

    def get_face_up(self):
        """Returns the face up cards for this player. Available to all players."""
        return self.face_up

    def number_down(self):
        """Returns the number of face down cards for this player. Available to all players."""
        return len(self.face_down)

    def is_out(self):
        """Returns true if this hand is empty (player is out), False otherwise."""
        if len(self.hand) > 0 or len(self.face_up) > 0 or len(self.face_down) > 0:
            return False
        return True


class Play(IntEnum):
    """Different possible play actions"""
    PICKED_UP = 1         # picked up the pile
    PLAYED_FROM_HAND = 2  # played from private hand
    PLAYED_FROM_UP = 3    # played from up cards
    PLAYED_FROM_DOWN = 4  # played from down cards


def prompt_cards(reason):
    """Asks the user to enter cards they want to play"""
    card_string = input(f"[For: {reason}] enter cards separated by commas: ")
    return np.array(card_string.split(','))


class Player:
    """A Jimmy's Game player. Default implementation outputs game state as it is told it, and prompts user to enter
    a selection string to make a move.

    Extend this class to create other players / agents with different functionality."""

    def assign_id(self, id):
        """Tell the player what their ID is - unique number indicating sequential play order."""
        self.id = id

    def assign_hand(self, hand: Hand):
        """Give this player their hand. Player should decide on their face up cards in this method."""
        self.hand = hand
        print(f"Initial hand: {self.hand.hand}")
        self.hand.decide_face_up(prompt_cards("deciding face up cards"))

    def take_cards(self, cards: np.ndarray):
        """Notify the player that they have taken cards from the draw pile"""
        return

    def state_change(self, id, play_type, played_cards, pile_before, pile_after):
        """Notify of a change in game state caused by the actions of any player, *including* ourselves."""
        print(
            f"---Player {id} played {played_cards} via {play_type}--- PILE: {pile_after}")

    def other_play(self, id, play_type, played_cards, pile, hands):
        """Notify this player of a change in game state caused by another players play. By default,
        nothing is done here, although agents may choose to record or use this information for their own purposes.

        Parameters:
        - id: ID of player who played
        - play_type: List of play(s) the player made (Play enum)
        - played_cards: List of cards played by user in each move (corresponds to play_type)
        - pile: Current contents of draw pile
        - hands: list of current hands for each player (only access face up and face down count)"""
        print(
            f"---Player {id} played {played_cards} via {play_type}--- PILE: {pile}")

    def your_play(self, hands, pile):
        """Request a play from this player. Provides hand information and pile information for convenience,
        although some player implementations may already store internally some of this information.

        Returns: (play_type, played_cards) tuple representing the play, and cards that are played.

        If the user plays cards that warrant another turn, then your_play will be called again by the environment."""
        print("""
            Play options
            PICKED_UP = 1         # picked up the pile
            PLAYED_FROM_HAND = 2  # played from private hand
            PLAYED_FROM_UP = 3    # played from up cards
            PLAYED_FROM_DOWN = 4  # played from down cards
            """)

        print(f"You are player {self.id}")
        print(
            f"Up cards: {self.hand.face_up}; Cards in hand: {self.hand.hand}; {self.hand.number_down()} down")
        hands_string = '\n'.join(
            [f'Player {i}: {hands[i]}' for i in range(len(hands))]
        )
        print(f"All hands: \n{hands_string}")
        print(f"Current pile: {pile}")
        play_type = Play(int(input("Enter play option: ")))
        if play_type == Play.PICKED_UP or play_type == Play.PLAYED_FROM_DOWN:
            return play_type, np.array([])

        played_cards = prompt_cards("cards to play this turn")

        # note: do not call any hand methods - environment does this to ensure hand accuracy
        return (play_type, played_cards)


def legal_play(hand: Hand, pile: np.array, play_type: Play, played_cards: np.array):
    """Confirm that the specified play is legal, based off of the cards in the players
    hand, the pile, and the play they made."""
    if play_type == Play.PICKED_UP:
        # this is only illegal if the pile is empty (no turn skipping)
        if len(pile) == 0:
            return False
        return True

    if play_type == Play.PLAYED_FROM_DOWN:
        # only illegal if player has cards in their hand, or face up still
        if len(hand.face_up) > 0 or len(hand.hand) > 0:
            return False
        return True

    if play_type == Play.PLAYED_FROM_UP:
        # illegal if hand is non-empty - has to be checked like hand cards though
        if len(hand.hand) > 0:
            return False

    # confirm valid play
    # confirm non-empty play
    if len(played_cards) == 0:
        return False

    # confirm all cards in play are identical
    if not np.isin(played_cards, played_cards[0]).all():
        return False

    # if pile is empty, this play is automatically valid
    if len(pile) == 0:
        return True

    # make sure that the card played is at least as high as the top card in the pile
    if not card_greater(played_cards[0], pile[-1]):
        return False

    # we're good! Valid play
    return True


def card_greater(a: str, b: str):
    """Returns true if a >= b, False otherwise."""
    if a == '2' or a == '10' or b == '2':  # always allowed
        return True
    idx_a = CARD_TYPES.index(a)
    idx_b = CARD_TYPES.index(b)
    return idx_a >= idx_b


class Environment:
    """Class that keeps track of an instance of a game of Jimmy's Game. Allows repeated plays,
    and handles interactions between players. Ensures that game rules are followed, keeping track of player
    hands and the deck as a whole."""

    def __init__(self):
        """Initialize internal data structures of environment"""
        # player management
        self.players: "list[Player]" = []  # index is player ID
        self.hands: "list[Hand]" = []  # index is player ID

        # game state management
        self.deck = None
        self.pile = np.array([])
        self.turn = -1  # ID of next player to play
        self.players_remaining = []  # list of indices of players in the game
        self.out_order = []  # list of indices as players get out

    def register_player(self, player: Player):
        """Add a player to this game of Jimmy's Game."""
        player_id = len(self.players)
        self.players.append(player)
        player.assign_id(player_id)

    def setup_game(self, num_decks=None):
        """Set up the game environment. This assigns users hands, which will in turn
        prompt them to assign face up cards. It also determines the first player to go 
        (first player sequentially with a 3, in this case)."""
        if num_decks is None:
            # if not specified, use minimum number of decks to give out cards
            num_decks = 1 + ((12 * len(self.players)) // 52)

        # init deck
        self.deck = Deck(num_decks)

        # deal out cards to players
        for i in range(len(self.players)):
            self.players_remaining.append(i)
            # create new hand
            player_hand = Hand(self.deck.draw(4), self.deck.draw(8))
            self.hands.append(player_hand)
            # player decides face up before this method returns
            self.players[i].assign_hand(player_hand)

        # find first sequential player with a 3 (then 4,5,6 etc), and give them the first turn
        current_value = 3
        while self.turn == -1:
            for i in range(len(self.hands)):
                if str(current_value) in self.hands[i].get_hand():
                    self.turn = i  # give this player ID first turn
                    break
            current_value += 1

    def advance_play(self):
        """Prompt the next player to play, and adjust the game state based off of their play."""
        # prompt player for their play
        (play_type, played_cards) = self.players[self.turn].your_play(
            self.hands, self.pile)

        self.last_play = (play_type, played_cards)  # this is used by other environment implementations

        # save current pile state before we modify it
        pile_before = self.pile

        # confirm play is legal
        if not legal_play(self.hands[self.turn], self.pile, play_type, played_cards):
            raise InvalidPlay(
                f"Player {self.turn} attempted to play {play_type}: {played_cards}")

        # adjust pile / player hands as needed

        # call appropriate hand play function based off of play
        if play_type == Play.PLAYED_FROM_HAND:
            self.hands[self.turn].play_from_hand(played_cards)
        elif play_type == Play.PLAYED_FROM_UP:
            self.hands[self.turn].play_face_up(played_cards)
        elif play_type == Play.PLAYED_FROM_DOWN:
            # special: we assign played_cards to this card
            played_cards = np.array([self.hands[self.turn].play_face_down()])

        # as long as the player played a card, append it to end of pile
        self.pile = np.append(self.pile, played_cards)

        if play_type == Play.PICKED_UP:
            # give them the pile!
            self.hands[self.turn].hand = np.append(
                self.hands[self.turn].hand, self.pile)
            self.pile = np.array([])
        elif play_type == Play.PLAYED_FROM_DOWN:
            # check if they got lucky (if pile size is one, no other cards to compare with)
            # we check against 2nd to last card in pile, since the down card is "on" the pile
            if len(self.pile) > 1 and not card_greater(played_cards[0], self.pile[-2]):
                # tough, they get the pile
                self.hands[self.turn].hand = np.append(
                    self.hands[self.turn].hand, self.pile)
                self.pile = np.array([])

        # notify other players of this play
        for i in range(len(self.players)):
            # if i == self.turn:  # needed for old calling system
            #     continue  # they already know, they played it
            # self.players[i].other_play(  # old calling convention
            #     self.turn, play_type, played_cards, self.pile, self.hands)

            # new notification system: state change
            self.players[i].state_change(self.turn, play_type, played_cards, pile_before, self.pile)

        # have player draw to 4 if possible
        if self.deck.left() > 0 and len(self.hands[self.turn].hand) < 4:
            needed_cards = 4 - len(self.hands[self.turn].hand)
            drawn_cards = self.deck.draw(needed_cards)
            self.hands[self.turn].hand = np.append(
                self.hands[self.turn].hand, drawn_cards)

            # notify player about drawn cards
            self.players[self.turn].take_cards(drawn_cards)

        # check if player is now out, modify data structures as needed, and increment turn
        if len(played_cards) >= 4 or (np.isin(played_cards, '10').any()):  # player played bomb
            bombed = True
        # bomb created with other cards in pile
        elif len(self.pile[-4:]) == 4 and np.isin(self.pile[-4:], self.pile[-1]).all():
            bombed = True
        else:
            bombed = False

        if bombed:  # clear pile
            self.pile = np.array([])

        if bombed and not self.hands[self.turn].is_out():
            return  # bombed, gets to play again

        # since we are here, know player either is out, or not out and bombed
        # both situations advance player
        old_turn = self.turn
        idx = self.players_remaining.index(self.turn)
        idx = (idx + 1) % len(self.players_remaining)
        self.turn = self.players_remaining[idx]

        # if player is out, remove them from players remaining
        if self.hands[old_turn].is_out():
            self.out_order.append(old_turn)
            self.players_remaining.remove(old_turn)

        # if only one player left, get them out of remaining queue too
        if len(self.players_remaining) == 1:
            self.out_order.append(self.turn)
            self.players_remaining.remove(self.turn)

    def play_game(self, num_decks=None):
        """Start the game of Jimmy's Game, optionally specifying a number of decks
        to play with. Once this is called, the environment will play through the game,
        prompting players as needed, until the game terminates."""
        self.setup_game(num_decks)
        while len(self.players_remaining) > 0:
            self.advance_play()
        return self.out_order


if __name__ == "__main__":
    # simple 2 player test
    player_1 = Player()
    player_2 = Player()
    env = Environment()
    env.register_player(player_1)
    env.register_player(player_2)
    env.play_game()
