"""
SECTION 1: FUZZY LOGIC PHQ-9 SCORER
======================================
Uses scikit-fuzzy to build a Mamdani-type fuzzy inference system
that scores PHQ-9 responses and outputs a depression severity level.

PHQ-9 Score Ranges:
  0–4   → Minimal depression
  5–9   → Mild depression
 10–14  → Moderate depression
 15–19  → Moderately severe depression
 20–27  → Severe depression
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyPHQ9Scorer:
    """
    Fuzzy Logic system for scoring PHQ-9 questionnaire answers.
    
    Each PHQ-9 item is answered on a scale of 0–3:
      0 = Not at all
      1 = Several days
      2 = More than half the days
      3 = Nearly every day
    
    The fuzzy system maps these 9 raw scores → overall severity score
    using membership functions and fuzzy rules.
    """

    def __init__(self, weights: list = None):
        """
        Args:
            weights: Optional list of 9 weights (from GA optimizer).
                     Defaults to uniform weights [1.0] * 9.
        """
        self.weights = weights if weights else [1.0] * 9
        self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        """Build membership functions and fuzzy rules."""

        # ── Universe of discourse ──────────────────────────────────────────
        # Single aggregated PHQ-9 input (weighted sum) ranges 0–27
        phq_score = ctrl.Antecedent(np.arange(0, 28, 0.1), 'phq_score')

        # Output: severity index 0–100
        severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

        # ── Membership Functions: Input ────────────────────────────────────
        phq_score['minimal']            = fuzz.trapmf(phq_score.universe, [0,  0,  3,  5])
        phq_score['mild']               = fuzz.trimf(phq_score.universe,  [4,  7,  10])
        phq_score['moderate']           = fuzz.trimf(phq_score.universe,  [9,  12, 15])
        phq_score['moderately_severe']  = fuzz.trimf(phq_score.universe,  [14, 17, 20])
        phq_score['severe']             = fuzz.trapmf(phq_score.universe, [19, 22, 27, 27])

        # ── Membership Functions: Output ───────────────────────────────────
        severity['minimal']            = fuzz.trapmf(severity.universe, [0,  0,  15, 25])
        severity['mild']               = fuzz.trimf(severity.universe,  [20, 35, 45])
        severity['moderate']           = fuzz.trimf(severity.universe,  [40, 55, 65])
        severity['moderately_severe']  = fuzz.trimf(severity.universe,  [60, 72, 82])
        severity['severe']             = fuzz.trapmf(severity.universe, [78, 88, 100, 100])

        # ── Fuzzy Rules ────────────────────────────────────────────────────
        rules = [
            ctrl.Rule(phq_score['minimal'],           severity['minimal']),
            ctrl.Rule(phq_score['mild'],              severity['mild']),
            ctrl.Rule(phq_score['moderate'],          severity['moderate']),
            ctrl.Rule(phq_score['moderately_severe'], severity['moderately_severe']),
            ctrl.Rule(phq_score['severe'],            severity['severe']),
        ]

        # Compound rules (OR combinations for boundary zones)
        rules += [
            ctrl.Rule(phq_score['minimal'] | phq_score['mild'],
                      severity['mild']),
            ctrl.Rule(phq_score['moderate'] | phq_score['moderately_severe'],
                      severity['moderately_severe']),
        ]

        self._ctrl_system = ctrl.ControlSystem(rules)
        self._simulator   = ctrl.ControlSystemSimulation(self._ctrl_system)

    def compute_weighted_score(self, responses: list) -> float:
        """
        Apply GA-optimized weights to raw PHQ-9 responses.

        Args:
            responses: List of 9 integers, each in [0, 3].
        Returns:
            Weighted score clipped to [0, 27].
        """
        if len(responses) != 9:
            raise ValueError("PHQ-9 requires exactly 9 responses.")

        # Normalise weights so they sum to 9 (preserve scale)
        w = np.array(self.weights, dtype=float)
        w = w / w.sum() * 9

        weighted = sum(r * wt for r, wt in zip(responses, w))
        return float(np.clip(weighted, 0, 27))

    def score(self, responses: list) -> dict:
        """
        Full fuzzy scoring pipeline.

        Args:
            responses: List of 9 integers [0–3].
        Returns:
            dict with keys:
              raw_score       – plain sum (0–27)
              weighted_score  – GA-weighted sum
              severity_index  – fuzzy output (0–100)
              severity_label  – human-readable label
              severity_pct    – severity_index as percentage string
        """
        raw_score      = sum(responses)
        weighted_score = self.compute_weighted_score(responses)

        # Feed into fuzzy system
        self._simulator.input['phq_score'] = weighted_score
        self._simulator.compute()
        severity_index = float(self._simulator.output['severity'])

        label = self._classify_label(weighted_score)

        return {
            "raw_score":      raw_score,
            "weighted_score": round(weighted_score, 2),
            "severity_index": round(severity_index, 2),
            "severity_label": label,
            "severity_pct":   f"{round(severity_index, 1)}%",
        }

    @staticmethod
    def _classify_label(score: float) -> str:
        if score <= 4:
            return "Minimal Depression"
        elif score <= 9:
            return "Mild Depression"
        elif score <= 14:
            return "Moderate Depression"
        elif score <= 19:
            return "Moderately Severe Depression"
        else:
            return "Severe Depression"

    def membership_degrees(self, weighted_score: float) -> dict:
        """
        Returns membership degrees of the weighted score for each category.
        Useful for explainability / UI visualisation.
        """
        universe = np.arange(0, 28, 0.1)
        mf_params = {
            "Minimal":           fuzz.trapmf(universe, [0,  0,  3,  5]),
            "Mild":              fuzz.trimf(universe,  [4,  7,  10]),
            "Moderate":          fuzz.trimf(universe,  [9,  12, 15]),
            "Moderately Severe": fuzz.trimf(universe,  [14, 17, 20]),
            "Severe":            fuzz.trapmf(universe, [19, 22, 27, 27]),
        }
        idx = (np.abs(universe - weighted_score)).argmin()
        return {k: round(float(v[idx]), 4) for k, v in mf_params.items()}


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = FuzzyPHQ9Scorer()

    test_cases = [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0], "Minimal expected"),
        ([1, 1, 1, 1, 1, 0, 0, 0, 0], "Mild expected"),
        ([2, 2, 1, 1, 2, 1, 1, 0, 1], "Moderate expected"),
        ([3, 3, 2, 2, 2, 2, 1, 1, 2], "Moderately Severe expected"),
        ([3, 3, 3, 3, 3, 3, 3, 3, 3], "Severe expected"),
    ]

    for responses, label in test_cases:
        result = scorer.score(responses)
        print(f"\n[{label}]")
        print(f"  Raw: {result['raw_score']}  |  Weighted: {result['weighted_score']}")
        print(f"  Severity Index: {result['severity_index']}  →  {result['severity_label']}")
        print(f"  Memberships: {scorer.membership_degrees(result['weighted_score'])}")