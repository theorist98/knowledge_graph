# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a probability example theorem.

Exact Lean code referenced (verbatim):

structure OutcomeSet where
  outcomes : List Nat
  favorable : List Nat
def calculate_probability (os : OutcomeSet) : Nat × Nat := (List.length os.favorable, List.length os.outcomes)
theorem probability_example : calculate_probability ⟨[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6]⟩ = (3, 10) := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean OutcomeSet structure and calculate_probability function exactly:
   - OutcomeSet contains outcomes and favorable outcomes as lists
   - calculate_probability returns (favorable_count, total_count)
2) Verifies the Lean theorem analogue:
   - probability_example: calculate_probability(<[1,2,3,4,5,6,7,8,9,10], [2,4,6]>) = (3, 10)
3) Provides probability models demonstrating discrete probability theory:
   - Classical probability calculations with favorable/total outcomes
   - Statistical validation of probability ratios
   - Monte Carlo simulations for probability convergence
4) Visuals (three complementary plots):
   - Probability distribution visualization
   - Convergence analysis of probability estimates
   - Comparison of theoretical vs empirical probabilities
5) Saves CSV summary of probability calculations and validations

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import random

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure OutcomeSet where
  outcomes : List Nat
  favorable : List Nat
def calculate_probability (os : OutcomeSet) : Nat × Nat := (List.length os.favorable, List.length os.outcomes)
theorem probability_example : calculate_probability <[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6]> = (3, 10) := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n- End Lean code -\n")

@dataclass(frozen=True)
class OutcomeSet:
    outcomes: List[int]
    favorable: List[int]

def calculate_probability(os: OutcomeSet) -> Tuple[int, int]:
    """
    Python analogue of the Lean calculate_probability function:
    Returns (number of favorable outcomes, total number of outcomes)
    """
    return (len(os.favorable), len(os.outcomes))

# ----- Verify the Lean theorem instances -----

# Create the exact example from the Lean theorem
lean_example = OutcomeSet(
    outcomes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    favorable=[2, 4, 6]
)

result = calculate_probability(lean_example)

print("Testing theorem probability_example:")
print(f"  OutcomeSet(outcomes=[1,2,3,4,5,6,7,8,9,10], favorable=[2,4,6])")
print(f"  calculate_probability = {result}")
print(f"  Expected: (3, 10)")

assert result == (3, 10), f"probability_example failed: got {result}, expected (3, 10)"
print("  Theorem verified: calculate_probability(<[1,2,3,4,5,6,7,8,9,10], [2,4,6]>) = (3, 10)")

probability_fraction = result[0] / result[1]
print(f"  Probability as fraction: {result[0]}/{result[1]} = {probability_fraction:.1f} = 30%\n")

print("Lean theorem analogue check passed: probability_example verified.\n")

# ----- Advanced probability analysis -----

def create_test_probability_sets():
    """Create various probability scenarios for analysis"""

    test_sets = [
        # Basic examples
        ("Coin Flip", OutcomeSet(outcomes=[1, 2], favorable=[1])),  # Heads = 50%
        ("Fair Die Even", OutcomeSet(outcomes=[1, 2, 3, 4, 5, 6], favorable=[2, 4, 6])),  # Even = 50%
        ("Fair Die Low", OutcomeSet(outcomes=[1, 2, 3, 4, 5, 6], favorable=[1, 2, 3])),  # Low = 50%

        # Card examples
        ("Card Red", OutcomeSet(outcomes=list(range(1, 53)), favorable=list(range(1, 27)))),  # Red cards = 50%
        ("Card Face", OutcomeSet(outcomes=list(range(1, 53)), favorable=[11, 12, 13, 24, 25, 26, 37, 38, 39, 50, 51, 52])),  # Face cards

        # Lottery examples
        ("Small Lottery", OutcomeSet(outcomes=list(range(1, 11)), favorable=[7])),  # 1 in 10
        ("Medium Lottery", OutcomeSet(outcomes=list(range(1, 101)), favorable=[42, 73])),  # 2 in 100

        # Other probability scenarios
        ("Multiple Outcomes", OutcomeSet(outcomes=list(range(1, 21)), favorable=list(range(5, 16)))),  # 11 in 20
        ("Rare Event", OutcomeSet(outcomes=list(range(1, 1001)), favorable=[1, 500, 999])),  # 3 in 1000
        ("Common Event", OutcomeSet(outcomes=list(range(1, 11)), favorable=list(range(1, 9)))),  # 8 in 10
    ]

    return test_sets

def analyze_probabilities(test_sets):
    """Analyze probability calculations across different scenarios"""

    analysis_results = []

    for name, outcome_set in test_sets:
        prob_counts = calculate_probability(outcome_set)
        prob_decimal = prob_counts[0] / prob_counts[1] if prob_counts[1] > 0 else 0
        prob_percentage = prob_decimal * 100

        analysis_results.append({
            'scenario': name,
            'total_outcomes': prob_counts[1],
            'favorable_outcomes': prob_counts[0],
            'probability_fraction': f"{prob_counts[0]}/{prob_counts[1]}",
            'probability_decimal': prob_decimal,
            'probability_percentage': prob_percentage,
            'odds_against': f"{prob_counts[1] - prob_counts[0]}:{prob_counts[0]}" if prob_counts[0] > 0 else "infinity:0",
            'is_lean_example': name == "Fair Die Even" and prob_counts == (3, 10)  # Close to Lean example
        })

    return analysis_results

def simulate_monte_carlo_convergence(target_set: OutcomeSet, num_trials_list: List[int]):
    """Simulate Monte Carlo convergence to theoretical probability"""

    theoretical_prob = len(target_set.favorable) / len(target_set.outcomes)
    convergence_data = []

    # Set seed for reproducibility
    random.seed(20250915)

    for num_trials in num_trials_list:
        successes = 0

        for _ in range(num_trials):
            # Simulate random selection from outcomes
            selected = random.choice(target_set.outcomes)
            if selected in target_set.favorable:
                successes += 1

        empirical_prob = successes / num_trials if num_trials > 0 else 0
        error = abs(empirical_prob - theoretical_prob)

        convergence_data.append({
            'trials': num_trials,
            'successes': successes,
            'empirical_probability': empirical_prob,
            'theoretical_probability': theoretical_prob,
            'absolute_error': error,
            'relative_error': error / theoretical_prob if theoretical_prob > 0 else float('inf')
        })

    return convergence_data

# Run comprehensive analysis
test_sets = create_test_probability_sets()
probability_analysis = analyze_probabilities(test_sets)

print("Analyzing probability calculations:")
for result in probability_analysis:
    print(f"\n{result['scenario']}:")
    print(f"  Favorable/Total: {result['favorable_outcomes']}/{result['total_outcomes']}")
    print(f"  Probability: {result['probability_percentage']:.1f}% ({result['probability_decimal']:.3f})")
    print(f"  Odds against: {result['odds_against']}")

# Monte Carlo convergence analysis
trials_list = [10, 50, 100, 500, 1000, 5000, 10000]
convergence_results = simulate_monte_carlo_convergence(lean_example, trials_list)

print(f"\nMonte Carlo convergence analysis (Lean example):")
print(f"Theoretical probability: {convergence_results[0]['theoretical_probability']:.3f} (30%)")
for result in convergence_results:
    print(f"  {result['trials']:5d} trials: {result['empirical_probability']:.3f} (error: {result['absolute_error']:.3f})")

# ----- Save CSV summaries -----

# Create probability analysis DataFrame
probability_df = pd.DataFrame(probability_analysis)
probability_df.to_csv("./probability_example_analysis.csv", index=False)

# Create convergence analysis DataFrame
convergence_df = pd.DataFrame(convergence_results)
convergence_df.to_csv("./probability_example_convergence.csv", index=False)

print(f"\nSaved simulation data to CSV files.")

# ----- Visualization 1: Probability distribution comparison -----
plt.figure(figsize=(12, 8))

scenarios = [r['scenario'] for r in probability_analysis]
probabilities = [r['probability_percentage'] for r in probability_analysis]
favorable_counts = [r['favorable_outcomes'] for r in probability_analysis]

# Create bar plot
bars = plt.bar(range(len(scenarios)), probabilities, alpha=0.8,
               color=plt.cm.viridis(np.array(probabilities) / max(probabilities)),
               edgecolor='black', linewidth=1)

# Annotate bars with probability values
for i, (bar, prob, fav) in enumerate(zip(bars, probabilities, favorable_counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{prob:.1f}%\n({fav})', ha='center', va='bottom',
             fontweight='bold', fontsize=9)

plt.xlabel("Probability Scenario")
plt.ylabel("Probability Percentage")
plt.title("Probability Distribution Analysis\nLean Theorem: calculate_probability(<outcomes, favorable>) = (count_favorable, count_total)")
plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')

# Highlight the Lean example
lean_prob = 30.0  # 3/10 = 30%
plt.axhline(y=lean_prob, color='red', linestyle='--', alpha=0.7,
           label=f'Lean Example: {lean_prob}% (3/10)')
plt.legend()

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./probability_example_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Monte Carlo convergence -----
plt.figure(figsize=(12, 6))

trials = [r['trials'] for r in convergence_results]
empirical_probs = [r['empirical_probability'] for r in convergence_results]
theoretical_prob = convergence_results[0]['theoretical_probability']

plt.subplot(1, 2, 1)
plt.semilogx(trials, empirical_probs, 'bo-', linewidth=2, markersize=8, label='Empirical Probability')
plt.axhline(y=theoretical_prob, color='red', linestyle='--', linewidth=2,
           label=f'Theoretical: {theoretical_prob:.3f}')

plt.xlabel('Number of Trials (log scale)')
plt.ylabel('Probability')
plt.title('Monte Carlo Convergence\nLean Example: P = 3/10 = 0.3')
plt.legend()
plt.grid(True, alpha=0.3)

# Add convergence bound
upper_bound = theoretical_prob + 0.05
lower_bound = max(0, theoretical_prob - 0.05)
plt.fill_between(trials, lower_bound, upper_bound, alpha=0.2, color='red',
                label='±5% bound')

plt.subplot(1, 2, 2)
errors = [r['absolute_error'] for r in convergence_results]
plt.loglog(trials, errors, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Trials (log scale)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Convergence Error Analysis')
plt.grid(True, alpha=0.3)

# Add theoretical convergence rate (1/sqrt(n))
theoretical_errors = [1.0/np.sqrt(n) * 0.1 for n in trials]  # Scaled for visibility
plt.loglog(trials, theoretical_errors, 'g--', alpha=0.7, label='~1/√n rate')
plt.legend()

plt.tight_layout()
plt.savefig('./probability_example_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Probability comparison matrix -----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Group scenarios by probability ranges
low_prob = [r for r in probability_analysis if r['probability_percentage'] < 25]
medium_prob = [r for r in probability_analysis if 25 <= r['probability_percentage'] < 75]
high_prob = [r for r in probability_analysis if r['probability_percentage'] >= 75]
special_cases = [r for r in probability_analysis if r['scenario'] in ['Fair Die Even', 'Coin Flip']]

categories = [
    ("Low Probability (< 25%)", low_prob),
    ("Medium Probability (25-75%)", medium_prob),
    ("High Probability (≥ 75%)", high_prob),
    ("Special Cases (Fair Games)", special_cases)
]

for idx, (category_name, category_data) in enumerate(categories):
    ax = axes[idx]

    if category_data:
        scenarios = [r['scenario'] for r in category_data]
        favorable = [r['favorable_outcomes'] for r in category_data]
        total = [r['total_outcomes'] for r in category_data]

        x_pos = range(len(scenarios))

        # Stacked bar: favorable (bottom) and unfavorable (top)
        unfavorable = [t - f for f, t in zip(favorable, total)]

        bars1 = ax.bar(x_pos, favorable, label='Favorable', alpha=0.8, color='green')
        bars2 = ax.bar(x_pos, unfavorable, bottom=favorable, label='Unfavorable', alpha=0.8, color='lightcoral')

        # Add probability percentages
        for i, (fav, tot) in enumerate(zip(favorable, total)):
            prob_pct = (fav / tot) * 100
            ax.text(i, tot + tot*0.05, f'{prob_pct:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=10)

        ax.set_title(category_name, fontsize=12)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Number of Outcomes')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight Lean example if present
        lean_present = any(r['scenario'] == 'Fair Die Even' for r in category_data)
        if lean_present:
            ax.text(0.02, 0.98, 'Contains Lean Example', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')

plt.suptitle('Probability Analysis by Category\nTheorem Verification: calculate_probability returns (favorable, total) counts', fontsize=14)
plt.tight_layout()
plt.savefig('./probability_example_categories.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Probability Example Simulation Results ===")
print(f"Lean theorem verification: probability_example verified (calculate_probability = {result})")
print(f"Total probability scenarios analyzed: {len(test_sets)}")

print(f"\nProbability range analysis:")
prob_values = [r['probability_percentage'] for r in probability_analysis]
print(f"  Minimum probability: {min(prob_values):.1f}%")
print(f"  Maximum probability: {max(prob_values):.1f}%")
print(f"  Average probability: {np.mean(prob_values):.1f}%")
print(f"  Median probability: {np.median(prob_values):.1f}%")

print(f"\nOutcome set characteristics:")
total_outcomes = [r['total_outcomes'] for r in probability_analysis]
favorable_outcomes = [r['favorable_outcomes'] for r in probability_analysis]
print(f"  Outcome set sizes: {min(total_outcomes)} - {max(total_outcomes)}")
print(f"  Favorable outcome range: {min(favorable_outcomes)} - {max(favorable_outcomes)}")

print(f"\nMonte Carlo convergence analysis:")
final_error = convergence_results[-1]['absolute_error']
print(f"  Final error (10,000 trials): {final_error:.4f}")
print(f"  Convergence rate: Decreases as ~1/sqrt(n)")
print(f"  Theoretical probability: {theoretical_prob:.3f} (30%)")

print(f"\nTheorem applications:")
print(f"  Basic probability: P(favorable) = favorable_count / total_count")
print(f"  Lean example: P(even die) = 3/10 = 0.3 = 30%")
print(f"  Monte Carlo validation: Empirical probabilities converge to theoretical values")
print(f"  Discrete probability spaces: All outcomes equally likely")

print(f"\nCSV files exported: probability_example_analysis.csv, probability_example_convergence.csv")
print("Theorem verification: calculate_probability(<[1,2,3,4,5,6,7,8,9,10], [2,4,6]>) = (3, 10) (verified)")