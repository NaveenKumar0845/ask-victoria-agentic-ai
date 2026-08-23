from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.graph import ask_victoria, route_intent
from src.guardrails import check_input


def main() -> None:
    dataset = pd.read_csv(Path("data/evaluation_queries.csv"))
    rows = []
    for _, row in dataset.iterrows():
        query = row["query"]
        expected_behavior = row["expected_behavior"]
        expected_intent = row.get("expected_intent", "")
        decision = check_input(query)

        if expected_behavior == "block":
            rows.append(
                {
                    "query": query,
                    "expected_behavior": expected_behavior,
                    "actual_behavior": "block" if not decision.allowed else "answer",
                    "intent_correct": True,
                    "behavior_correct": not decision.allowed,
                    "groundedness": None,
                }
            )
            continue

        result = ask_victoria(query)
        predicted = result.get("intent", route_intent(query))
        rows.append(
            {
                "query": query,
                "expected_behavior": expected_behavior,
                "actual_behavior": "block" if result.get("blocked") else "answer",
                "intent_correct": predicted == expected_intent,
                "behavior_correct": not result.get("blocked", False),
                "groundedness": result.get("groundedness_score", 0.0),
            }
        )

    results = pd.DataFrame(rows)
    out = Path("data/evaluation_results.csv")
    results.to_csv(out, index=False)
    print(results.to_string(index=False))
    print()
    print(f"Intent accuracy: {results['intent_correct'].mean():.1%}")
    print(f"Behavior accuracy: {results['behavior_correct'].mean():.1%}")
    answered = results[results["groundedness"].notna()]
    if len(answered):
        print(f"Mean groundedness proxy: {answered['groundedness'].astype(float).mean():.1%}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
