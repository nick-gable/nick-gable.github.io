use pyo3::prelude::*;
mod mcts;
use mcts::Card;

/// Python bridge to search_best_play in MCTS module.
///
/// ### Parameters
/// - our_hand_up: Vec<String>,
/// - our_up_cards: Vec<String>,
/// - our_down_count: u8,
/// - opp_hand_up: Vec<String>,
/// - opp_up_cards: Vec<String>,
/// - opp_down_count: u8,
/// - down_cards: Vec<String>,
/// - pile: Vec<String>,
/// - our_turn: bool,
/// - times: u32,
/// - debug: bool
///
/// Returns:
/// (int, list[string]) representing play choice. int matches play type from Play enum
/// **in Python**.
#[pyfunction]
fn search_best_play(
    our_hand_up: Vec<String>,
    our_up_cards: Vec<String>,
    our_down_count: u8,
    opp_hand_up: Vec<String>,
    opp_up_cards: Vec<String>,
    opp_down_count: u8,
    down_cards: Vec<String>,
    pile: Vec<String>,
    our_turn: bool,
    times: u32,
    debug: bool,
) -> PyResult<(u8, Vec<String>)> {
    // Set up our state struct
    let state = mcts::State {
        our_hand: (
            card_from_python(our_hand_up),
            card_from_python(our_up_cards),
            our_down_count,
        ),
        opponent_hand: (
            card_from_python(opp_hand_up),
            card_from_python(opp_up_cards),
            opp_down_count,
        ),
        down_cards: card_from_python(down_cards),
        pile: card_from_python(pile),
        our_turn,
    };

    // Call Rust search_best_play
    let result = mcts::search_best_play(state, times, debug);

    // Convert play back into something Python understands
    // number return codes are for Python Play enum
    match result {
        mcts::Play::PickUp => Ok((1, Vec::new())),
        mcts::Play::PlayFromDown => Ok((4, Vec::new())),
        mcts::Play::PlayFromHand(card, num) => Ok((2, card_to_python(card, num))),
        mcts::Play::PlayFromUp(card, num) => Ok((3, card_to_python(card, num))),
    }
}

/// Function to convert a vector of string cards into internal card enum
fn card_from_python(card_vec: Vec<String>) -> Vec<mcts::Card> {
    let mut result: Vec<mcts::Card> = Vec::new();

    for card in card_vec {
        result.push(match card.as_str() {
            "3" => Card::Three,
            "4" => Card::Four,
            "5" => Card::Five,
            "6" => Card::Six,
            "7" => Card::Seven,
            "8" => Card::Eight,
            "9" => Card::Nine,
            "J" => Card::Jack,
            "Q" => Card::Queen,
            "K" => Card::King,
            "A" => Card::Ace,
            "2" => Card::Two,
            "10" => Card::Ten,
            _ => {
                panic!("Invalid card string")
            }
        });
    }

    result
}

/// Convert vector of Card enums back to Python list
fn card_to_python(card: Card, num: u8) -> Vec<String> {
    let card_str = match card {
        Card::Three => "3".to_string(),
        Card::Four => "4".to_string(),
        Card::Five => "5".to_string(),
        Card::Six => "6".to_string(),
        Card::Seven => "7".to_string(),
        Card::Eight => "8".to_string(),
        Card::Nine => "9".to_string(),
        Card::Jack => "J".to_string(),
        Card::Queen => "Q".to_string(),
        Card::King => "K".to_string(),
        Card::Ace => "A".to_string(),
        Card::Two => "2".to_string(),
        Card::Ten => "10".to_string(),
    };

    vec![card_str; num.into()]
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn jimmymcts(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_best_play, m)?)?;
    Ok(())
}
