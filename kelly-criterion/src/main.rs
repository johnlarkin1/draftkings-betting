/// This package is going to just compute the Kelley Criterion
/// for a given bank roll and a given set of odds.
use std::io;
use std::io::Write;

fn get_input(prompt: &str, default: String) -> String {
    print!("{}: ", prompt);
    io::stdout().flush().unwrap();
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    if buffer.trim().is_empty() {
        default
    } else {
        buffer
    }
}

fn compute_probability(sign: &str, american_odds: f32) -> f32 {
    match sign {
        "+" => 100.0 / (american_odds + 100.0),
        "-" => american_odds / (100.0 + american_odds),
        _ => {
            panic!("The inputed American odds must start with + or -");
        }
    }
}

fn compute_gain_proportion(sign: &str, american_odds: f32) -> f32 {
    match sign {
        "+" => american_odds / 100.0,
        "-" => 100.0 / american_odds,
        _ => {
            panic!("The inputed American odds must start with + or -");
        }
    }
}

fn compute_prob_and_payout(odds: &str) -> (f32, f32) {
    let (sign, str_digits) = odds.split_at(1);
    let odds = str_digits.trim().parse::<f32>().ok().unwrap();
    (
        compute_probability(sign, odds),
        compute_gain_proportion(sign, odds),
    )
}

fn compute_kelly(bankroll: f32, odds: &str, mult: f32) -> f32 {
    let (_, gain) = compute_prob_and_payout(odds);
    let prob = 0.51;
    println!("Prob: {} Gain: {}", prob, gain);
    let kelly_proportion = prob - ((1.0 - prob) / gain);
    kelly_proportion * bankroll * mult
}

fn main() {
    let multiplier = get_input("Enter your Kelly multiplier [0.5]", String::from("0.5"));
    let bankroll = get_input("Enter your total bankroll [500]", String::from("500"));
    let odds = get_input(
        "Enter the American odds (include +/-)",
        String::from("+125"),
    );

    // Kelly Criterion says
    // f* = p - (1 - p) / b
    // where f* is fraction to bet
    // p is probability of bet hitting
    // b is proportion of the bet gained
    let mult: f32 = multiplier.trim().parse().unwrap();
    let bank: f32 = bankroll.trim().parse().unwrap();
    let amount_to_wager = compute_kelly(bank, &odds, mult);
    println!("Computed amount to wager: {}", amount_to_wager);
}
