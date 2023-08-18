//! Efficiently use Monte-Carlo Tree Search to determine the best possible play for a player of Jimmy's Game
//!
//! Author: Nick Gable (gable105@umn.edu)

use rand::Rng;
use std::{cmp::Ordering, collections::HashMap, time::Instant};

const EXPLORATION_CONSTANT: f32 = 1.5;

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy, PartialOrd, Ord)]
pub enum Card {
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Jack,
    Queen,
    King,
    Ace,
    Two,
    Ten,
}

/// Representation of game state. Used to index transposition table
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct State {
    pub our_hand: (Vec<Card>, Vec<Card>, u8),
    pub opponent_hand: (Vec<Card>, Vec<Card>, u8),
    /// Vector of possible down cards (needs to be randomly drawn from!)
    pub down_cards: Vec<Card>,
    pub pile: Vec<Card>,
    /// Bool representing if it is currently our turn, or our opponents
    pub our_turn: bool,
}

/// Enum representing possible plays
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Play {
    PlayFromHand(Card, u8),
    PlayFromUp(Card, u8),
    PlayFromDown,
    PickUp,
}

/// Using Monte-Carlo Tree Search, find and return the best course
/// of action for the agent. Designed to be used during later stages of the game
/// when the game state is very well known by an agent who is tracking the cards,
/// save for face down cards (which can still be modelled using probabilities of
/// remaining cards).
///
/// ## Parameters
/// - `our_hand` is representation of our hand, consisting of a 3-tuple consisting of
///   a) the cards in our private hand; b) the cards we have face up; c) number of
///   cards face down.
/// - `opponent_hand` is the same representation as `our_hand`. Note that this obviously
///   requires that the Python agent is keeping track of cards enough to know this.
/// - `down_cards` is a vector of possible down cards for *either* player - this should be
///   randomly drawn from when the AI needs to simulate a random card draw.
/// - `pile` is a vector of cards representing the current active game pile.
/// - `times` number of times to sample the game tree.
///
/// ## Returns
/// A Play enum representing the optimal play choice.
///
/// ## Notes
/// - All hands should be *sorted* before they are passed into this function. This is to ensure
///   internal consistency when they are loaded into the transposition table.
pub fn search_best_play(start_state: State, times: u32, debug: bool) -> Play {
    // Initialize transposition table (key=state, val=(wins, times, parent_times))
    let mut table: HashMap<State, (u32, u32, u32)> = HashMap::new();
    //println!("Start state fed to me: {:?}", start_state);

    // Record start time - will break after two seconds
    let start_time = Instant::now();

    // Repeatedly probe tree with MCTS (fixed amount, or until best play converges)
    'outer: for round in 1..(times + 1) {
        // before starting loop: if we have elapsed too long, break early
        if start_time.elapsed().as_secs_f32() > 2.0 {
            if debug {
                println!("Ending early after {} iters", round);
            }
            break;
        }

        let mut current_state = start_state.clone();
        let mut states: Vec<State> = vec![start_state.clone()];
        let cycled: bool;
        // Main MCTS loop: continually update game state, keeping track of plays, until someone wins
        loop {
            // simulate the random play to update game state
            let best_play = find_best_uct_play(
                &current_state,
                &table,
                states.len().try_into().unwrap(),
                debug,
                false,
            );
            if debug {
                println!("Playing {:?}, state {:?}", best_play, current_state);
            }

            let new_state = simulate_play(&current_state, best_play);

            // check if we are at terminating condition - if so, break loop
            let mut terminate =
                player_won(&new_state.our_hand) || player_won(&new_state.opponent_hand);
            current_state = new_state.clone();
            states.push(new_state);

            if states.len() > 100 {
                // this state is cycling - skip it and start over from the beginning
                // TODO: Without cycle breaking, cycles may happen very frequently, which will reduce accuracy
                if debug {
                    println!("CYCLE - skipping this iteration");
                }
                cycled = true;
                break;
            }

            if terminate {
                // We are done!!
                if debug {
                    println!("---Game done---");
                }
                cycled = false;
                break;
            }
        }

        // Determine if we won
        let we_won = player_won(&current_state.our_hand);

        // Run through each play, updating transposition table before starting next loop
        let mut previous_plays = round;
        for state in states {
            // NOTE: We match !we_won since states store the result of the decision made by the
            // *previous* player, *not* the current one
            let win_value = match (!we_won, state.our_turn) {
                (true, true) | (false, false) => {
                    // only add point if we won, and it is our turn
                    // note: if we cycled, always record a loss to dissuade that path
                    if !cycled {
                        1
                    } else {
                        0
                    }
                }
                _ => 0,
            };

            match table.get_mut(&state) {
                None => {
                    // new table item, set default values and continue
                    table.insert(state, (win_value, 1, previous_plays));
                    previous_plays = 1; // we were only played once
                }
                Some((wins, plays, prev_plays)) => {
                    // Update existing values
                    *wins += win_value;
                    *plays += 1;
                    *prev_plays += 1;
                }
            }
        }
    }

    if debug {
        println!("Table at end of game: {:?}", table);
    }

    // Find and return the best play from the top-level
    find_best_uct_play(&start_state, &table, 0, debug, true)
}

/// Finds and returns the best (by UCT) play, given the transposition table and current state
/// ## Parameters
/// - `current_state` current state as state reference
/// - `table` transposition table
/// - `depth` depth we are searching at - used for heuristics to save time
/// - `debug` debug flag
/// - `for_soln` compute best play for what we return to Python
fn find_best_uct_play(
    current_state: &State,
    table: &HashMap<State, (u32, u32, u32)>,
    depth: u32,
    debug: bool,
    for_soln: bool,
) -> Play {
    // Assemble possible plays to create our branch options
    let (hand, up, _) = if current_state.our_turn {
        &current_state.our_hand
    } else {
        &current_state.opponent_hand
    };

    // this option, plus picking up (assuming non-empty pile)
    let play_option = if hand.len() == 0 && up.len() == 0 {
        Play::PlayFromDown
    } else if hand.len() == 0 {
        Play::PlayFromUp(Card::Ace, 1) // dummy interior values
    } else {
        Play::PlayFromHand(Card::Ace, 1)
    };

    let mut branches = match play_option {
        Play::PlayFromDown => {
            // Only option is to play down card
            vec![Play::PlayFromDown]
        }
        Play::PlayFromUp(_, _) => {
            // iterate through all up card options
            let plays = possible_plays(up);
            plays
                .iter()
                .map(|(c, n)| Play::PlayFromUp(*c, *n))
                .collect()
        }
        Play::PlayFromHand(_, _) => {
            // similar to above, iterate through hand options
            let plays = possible_plays(hand);
            plays
                .iter()
                .map(|(c, n)| Play::PlayFromHand(*c, *n))
                .collect()
        }
        _ => panic!("Strange play option match in loop"),
    };

    // prune out branches that are illegal
    let mut i = 0;
    let top_card = current_state.pile.last();
    while i < branches.len() && top_card != None {
        let branch = branches.get(i).unwrap();
        match branch {
            Play::PlayFromUp(card, _) | Play::PlayFromHand(card, _) => {
                // need to check that this card value is at least as high as top card
                if *card == Card::Two || *card == Card::Ten || top_card.unwrap() == &Card::Two {
                    // we can always play, so continue
                    i += 1;
                } else if card >= top_card.unwrap() {
                    // our card is greater
                    i += 1;
                } else {
                    // remove this branch, not a valid play
                    branches.remove(i);
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // add picking up to branch option list
    if current_state.pile.len() > 0 {
        branches.push(Play::PickUp);
    }

    // compute current node visits (used for parent node calculation)
    // if match doesn't work, all nodes will be unvisited, so no need to use the value
    let current_node_visits = match table.get(current_state) {
        Some((_, times, _)) => *times,
        None => 1,
    };

    // compute the UCT for each branch, and decide best option
    let mut uct_vals: Vec<f32> = Vec::new();
    let mut unexplored_branches: Vec<Play> = Vec::new();

    for play in branches.iter() {
        match play {
            Play::PlayFromDown => {
                // We don't simulate this because it just needs to be done
                // doing this jumps to a random state, so we may need to account for this
                uct_vals.push(f32::INFINITY);
                break;
            }
            _ => {
                // every other option is deterministic, want to actually compute
                let next_state = simulate_play(&current_state, *play);
                match table.get(&next_state) {
                    Some((wins, times, p_times)) => {
                        if for_soln {
                            // if used for solution, use denominator instead
                            uct_vals.push((*times as f32));
                        } else {
                            uct_vals.push(uct(*wins, *times, current_node_visits));
                        }
                    }
                    None => {
                        // randomly generate weight for unvisited node
                        // random generated will be higher than other nodes on purpose, so they always get picked
                        // let mut rng = rand::thread_rng();
                        // uct_vals.push(5.0 + rng.gen::<f32>());

                        // we are exploring: mark that as true and break
                        unexplored_branches.push(*play);
                    }
                }
            }
        }
    }

    if unexplored_branches.len() > 0 {
        // we are exploring - want to follow heuristic of exploring the lowest playable cards first
        // (generally the best strategy, which will save evaluation time)
        if debug {
            println!("Unexplored: Using heuristic");
        }

        let mut branches = unexplored_branches;
        branches.sort(); // sorting branches will put playing cards first before picking up, and sort values lower too

        // want to play as many of our lowest card as we can
        match branches.get(0).unwrap() {
            Play::PickUp => {
                return Play::PickUp;
            }
            Play::PlayFromHand(card, _) | Play::PlayFromUp(card, _) => {
                // this fancy expression basically returns the branch that plays the maximum number of cards
                // while still keeping the card value at a minimum
                return *branches
                    .iter()
                    .filter(|x| match x {
                        Play::PlayFromHand(c, _) | Play::PlayFromUp(c, _) => c == card,
                        _ => false,
                    })
                    .last()
                    .unwrap();
            }
            Play::PlayFromDown => {
                return Play::PlayFromDown;
            }
        }
    }

    let best_idx = uct_vals
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .expect("Unable to find best play")
        .0;

    if debug {
        print!("UCT: ");
        for i in 0..uct_vals.len() {
            print!(
                "({:?},{:.5}),",
                branches.get(i).unwrap(),
                uct_vals.get(i).unwrap()
            );
        }
        
        println!();
    }

    let best_play = branches.get(best_idx).unwrap();

    *best_play
}

/// Simulate a single play, and return the new game state.
///
/// ## Parameters
/// - `state` current game state
/// - `play` chosen play
/// - `our_turn` bool representing whether this play is being done by us or our opponent
///
/// ## Returns
/// New State struct with the play change propagated
///
/// ## Notes
/// Modified hands are returned sorted so that ordering is consistent for the transposition table.
pub fn simulate_play(state: &State, play: Play) -> State {
    let mut new_state = state.clone();
    match play {
        Play::PickUp => {
            // Give the player in question all the cards in the draw pile
            if state.our_turn {
                // We get the draw pile :(
                new_state.our_hand.0.append(&mut new_state.pile);
                new_state.our_hand.0.sort();
            } else {
                // Haha, opponent gets draw pile
                new_state.opponent_hand.0.append(&mut new_state.pile);
                new_state.opponent_hand.0.sort();
            }
        }
        Play::PlayFromDown => {
            // Randomly draw from the possible down cards, and either add it to the pile,
            // or give it to the player who played plus the pile if they lose
            let mut rng = rand::thread_rng();
            let card = new_state
                .down_cards
                .remove(rng.gen::<usize>() % new_state.down_cards.len());

            // determine if card warrants a pickup, or not
            let mut card_greater = false;
            match state.pile.last() {
                Some(top_card) => {
                    if card == Card::Two || card == Card::Ten || top_card == &Card::Two {
                        card_greater = true;
                    } else if card >= *top_card {
                        card_greater = true;
                    }
                }
                None => {
                    // empty pile, card always greater
                    card_greater = true;
                }
            }

            // get current players hand, decrement their counter
            let (hand, _, down) = if state.our_turn {
                &mut new_state.our_hand
            } else {
                &mut new_state.opponent_hand
            };

            *down -= 1;

            // if card is greater, add card to pile; else, give to player
            if card_greater {
                new_state.pile.push(card);
            } else {
                hand.push(card);
            }
        }
        Play::PlayFromHand(card, num) => {
            // Remove cards from players hand, and place them in the pile
            let (hand, _, _) = if state.our_turn {
                &mut new_state.our_hand
            } else {
                &mut new_state.opponent_hand
            };
            let mut count: u8 = 0;
            let mut removed: Vec<Card> = Vec::new();
            let mut i = 0;
            while i < hand.len() && count < num {
                if hand.get(i).unwrap() == &card {
                    count += 1;
                    removed.push(hand.remove(i));
                } else {
                    i += 1;
                }
            }

            if count != num {
                println!("Trying to play {:#?} with state {:#?}", play, state);
                assert_eq!(count, num);
            }

            new_state.pile.append(&mut removed);
        }
        Play::PlayFromUp(card, num) => {
            // Remove cards from players up cards, and place them in the pile
            let (_, hand, _) = if state.our_turn {
                &mut new_state.our_hand
            } else {
                &mut new_state.opponent_hand
            };
            let mut count: u8 = 0;
            let mut removed: Vec<Card> = Vec::new();
            let mut i = 0;
            while i < hand.len() && count < num {
                if hand.get(i).unwrap() == &card {
                    count += 1;
                    removed.push(hand.remove(i));
                } else {
                    i += 1;
                }
            }

            if count != num {
                println!("Trying to play {:?} with state {:?}", play, state);
                println!("Final state: {:?}", new_state);
                assert_eq!(count, num);
            }

            new_state.pile.append(&mut removed);
        }
    }

    new_state.our_turn = if state.our_turn { false } else { true };

    // If a bomb has happened, flip back
    if new_state.pile.len() > 0 && *new_state.pile.last().unwrap() == Card::Ten {
        new_state.our_turn = if state.our_turn { false } else { true };
    } else if new_state.pile.len() >= 4 {
        let last_four: Vec<&Card> = new_state.pile.iter().rev().take(4).collect();
        if last_four
            .iter()
            .all(|x| *x == new_state.pile.last().unwrap())
        {
            // last four cards are all equal, also a bomb
            new_state.our_turn = if state.our_turn { false } else { true };
        }
    }
    new_state
}

/// Calculate Upper Confidence Bound used to determine which path to take
///
/// ## Parameters
/// - `wins` number of wins at this node
/// - `times` number of times this node has been played
///
/// ## Returns
/// f32 representing UCT value.
fn uct(wins: u32, times: u32, parent_times: u32) -> f32 {
    let wins = wins as f32;
    let times = times as f32;
    let parent_times = parent_times as f32;
    (wins / times) + EXPLORATION_CONSTANT * (2.0 * parent_times.ln() / times).sqrt()
}

/// Iterate through and return the possible plays, given a hand.
///
/// ## Parameters
/// - `hand` vector of cards representing hand (where the player has it does not matter)
///
/// ## Returns
/// A vector of (Card, u8) pairs representing possible plays (card, and number of plays)
fn possible_plays(hand: &Vec<Card>) -> Vec<(Card, u8)> {
    let mut card_map: HashMap<Card, u8> = HashMap::new();
    let mut plays: Vec<(Card, u8)> = Vec::new();
    for card in hand {
        match card_map.get_mut(card) {
            Some(val) => {
                // increment counter
                *val += 1;
                // add play
                plays.push((*card, *val));
            }
            None => {
                // add to hash map
                card_map.insert(*card, 1);
                plays.push((*card, 1));
            }
        }
    }

    plays
}

/// Checks if a hand has won
fn player_won(hand: &(Vec<Card>, Vec<Card>, u8)) -> bool {
    let (private_hand, up, down_count) = hand;
    private_hand.len() == 0 && up.len() == 0 && *down_count == 0
}

/// Print out current MCTS state information for debugging
fn dump_state(current_state: &State, states: &Vec<State>, table: &HashMap<State, (u32, u32, u32)>) {
    println!("Current state: {:#?}", current_state);
    println!("All states: {:#?}", states);
    println!("Transposition table: {:#?}", table);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::Card::*; // makes tests cleaner

    #[test]
    fn test_possible_plays() {
        // TODO: This test may fail if ordering is off - this assumes plays is ordered like hands
        // may need to change test if this is not the case in actual implementation
        let hands = [
            vec![Three, Four, Five],
            vec![Three, Four, Five, Six, Three, Three],
            vec![Ten, Ten, Two, Five, Six],
            vec![Ace, King, Ace, King, Two],
        ];
        let plays = [
            vec![(Three, 1), (Four, 1), (Five, 1)],
            vec![
                (Three, 1),
                (Four, 1),
                (Five, 1),
                (Six, 1),
                (Three, 2),
                (Three, 3),
            ],
            vec![(Ten, 1), (Ten, 2), (Two, 1), (Five, 1), (Six, 1)],
            vec![(Ace, 1), (King, 1), (Ace, 2), (King, 2), (Two, 1)],
        ];

        for i in 0..hands.len() {
            assert_eq!(possible_plays(&hands[i]), plays[i]);
        }
    }

    #[test]
    fn test_uct() {
        let tests: [(u32, u32, u32); 4] = [(3, 5, 6), (3, 7, 10), (15, 80, 400), (0, 1, 1)];

        let correct: [f32; 4] = [1.446584, 1.23967, 0.5745228, 0.0];

        for i in 0..tests.len() {
            let (a, b, c) = tests[i];
            let actual = uct(a, b, c);
            assert!((actual - correct[i]).abs() < 0.00001);
        }
    }

    #[test]
    fn test_simulate_our_pickup() {
        let state = State {
            our_hand: (
                vec![Three, Four, Five, Five, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![Three, Five, Five, Six],
        };

        let after_play = State {
            our_hand: (
                vec![Three, Three, Four, Five, Five, Five, Five, Six, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![],
        };

        let result = simulate_play(&state, Play::PickUp);
        assert_eq!(result, after_play);
    }

    #[test]
    fn test_simulate_opponent_pickup() {
        let state = State {
            our_hand: (
                vec![Three, Four, Five, Five, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![Three, Five, Five, Six],
        };

        let after_play = State {
            our_hand: (
                vec![Three, Four, Five, Five, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            opponent_hand: (
                vec![Three, Four, Four, Five, Five, Five, Six, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![],
        };

        let result = simulate_play(&state, Play::PickUp);
        assert_eq!(result, after_play);
    }

    #[test]
    fn test_play_from_our_hand() {
        let state = State {
            our_hand: (
                vec![Three, Four, Five, Five, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![Three, Five],
        };

        let after_play = State {
            our_hand: (vec![Three, Four, Six], vec![Jack, Ace, Two, Two], 4),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![Three, Five, Five, Five],
        };

        let result = simulate_play(&state, Play::PlayFromHand(Five, 2));
        assert_eq!(result, after_play);
    }

    #[test]
    fn test_play_from_opponent_hand() {
        let state = State {
            opponent_hand: (
                vec![Three, Four, Five, Five, Six],
                vec![Jack, Ace, Two, Two],
                4,
            ),
            our_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![Three, Five],
        };

        let after_play = State {
            opponent_hand: (vec![Three, Four, Six], vec![Jack, Ace, Two, Two], 4),
            our_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![Three, Five, Five, Five],
        };

        let result = simulate_play(&state, Play::PlayFromHand(Five, 2));
        assert_eq!(result, after_play);
    }

    #[test]
    fn test_play_from_our_up() {
        let state = State {
            our_hand: (vec![], vec![Jack, Ace, Two, Two], 4),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![Three, Five],
        };

        let after_play = State {
            our_hand: (vec![], vec![Jack, Ace], 4),
            opponent_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![Three, Five, Two, Two],
        };

        let result = simulate_play(&state, Play::PlayFromUp(Two, 2));
        assert_eq!(result, after_play);
    }

    #[test]
    fn test_play_from_opponent_up() {
        let state = State {
            opponent_hand: (vec![], vec![Jack, Ace, Two, Two], 4),
            our_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: false,
            pile: vec![Three, Five],
        };

        let after_play = State {
            opponent_hand: (vec![], vec![Jack, Ace], 4),
            our_hand: (
                vec![Four, Four, Five, Six, Ace],
                vec![Seven, Seven, Ten, Ten],
                4,
            ),
            down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
            our_turn: true,
            pile: vec![Three, Five, Two, Two],
        };

        let result = simulate_play(&state, Play::PlayFromUp(Two, 2));
        assert_eq!(result, after_play);
    }
}
