//! Main program used for some testing that may not unit test well
pub mod mcts;
use mcts::{search_best_play, Play};

use crate::mcts::Card::*;
fn main() {
    // Test functionality of MCTS to determine best play
    // this is an interesting one - it doesn't converge to the same value every time
    let state_old = mcts::State {
        our_hand: (vec![Five, Eight], vec![], 3),
        opponent_hand: (vec![], vec![], 2),
        down_cards: vec![Seven, Six, Five, Five, Eight],
        our_turn: true,
        pile: vec![],
    };

    let state = mcts::State {
        our_hand: (
            vec![Three, Three, Five, Seven, Nine, Jack, Queen, King, Ace],
            vec![],
            0,
        ),
        opponent_hand: (vec![Three, Six, Eight, Queen, Queen, King], vec![], 0),
        down_cards: vec![],
        our_turn: true,
        pile: vec![Ace],
    };

    let stuck_state = mcts::State {
        our_hand: (
            vec![Eight, Jack, Jack, Queen, Queen, Queen, King],
            vec![],
            0,
        ),
        opponent_hand: (vec![Two, Two], vec![Eight, Five, Five, Four], 4),
        down_cards: vec![Six, Four, Six, Ten],
        our_turn: true,
        pile: vec![King, King],
    };

    loop {
        let debug = true;
        let best_play = search_best_play(state.clone(), 10000, debug);
        println!("{:?}", best_play);
        if debug {
            break;
        }
    }
}

fn rand_draw_test() {
    let state = mcts::State {
        opponent_hand: (vec![], vec![Jack, Ace, Two, Two], 4),
        our_hand: (
            vec![Four, Four, Five, Six, Ace],
            vec![Seven, Seven, Ten, Ten],
            4,
        ),
        down_cards: vec![Seven, Seven, Seven, Four, Six, Eight, Two, Five],
        our_turn: true,
        pile: vec![Three, Five],
    };

    let result = mcts::simulate_play(&state, Play::PlayFromDown);
    println!("{:?}", state);
    println!("{:?}", result);
}
