# JimmAI - Jimmy's Game AI
Hello! This is my Jimmy's Game AI repository, stripped of school content. My documentation here isn't exhaustive, but you should find that every file is (generally) very well documented. 

At a high level, the Python modules `environment`, `naive`, and `advanced` contain most of the important Python code for simulating the game environment. The `mcts` directory contains the rust library that powers the MCTS computations, which are done when there are only two players remaining. The remaining Python modules are used for testing, analyzing, and generating results that were used for the paper. Check them out if you want!

Unfortunately, I never created a requirements file for this project, so running this yourself might be a little challenging, but generally you should follow the process for building an application using PyO3 (for integrating Rust and Python) and make sure that `numpy` and `tqdm` are installed, plus `matplotlib` if you want to graph results.

## Environment implementation details
### `Deck`
Class that implements a deck of arbitrary number of 52 card decks. Only handles initial creation / shuffle and drawing from deck.

### `Hand`
Used to store hand information. Stores face up, face down, and cards in hand, and allows players to play from each location. Also has getter methods so that other players can access hand information. Only performs checks to ensure that players are not trying to play cards that are not in their hand (no game rules are enforced here, only those to ensure logical card manipulation).

Note: Hands are referenced by both the players and the environment. However, _hands are always modified by the environment_, except for when the player initially decides their face up cards. 

### `Play`
Enum that specifies what play a user did - options are:
```
PICKED_UP = 1         # picked up the pile
PLAYED_FROM_HAND = 2  # played from private hand
PLAYED_FROM_UP = 3    # played from up cards
PLAYED_FROM_DOWN = 4  # played from down cards
```

### `Player`
Representation of a game player. Gets overridden to create actual agents. Players are assigned IDs and hands, notified when other players play, and asked for a play once it is their turn. This class also does not enforce game mechanics.

### `Environment`
Core environment that keeps track of a game of Jimmy's Game. Takes in players and game information, and assigns them hands. Essentially acts as a dealer / overseer that keeps track of game play, prompting users to play, notifying other users, and ensuring no rules are broken (the only class that does this).

## Player-Environment API
A more advanced API is needed since the current one is convoluted and makes it difficult for agents to keep track of game state.

Our environment will continue to update the player hands for them, though.

```python
# environment calls this on every player after every turn,
# **including** their own. Makes it easier to track cards and plays
def state_change(id, play_type, played_cards, pile_before, pile_after)

# notifies player of cards they are given from the draw pile
# this is so the agent can remove them from counts if they desire
def take_cards(cards)
```