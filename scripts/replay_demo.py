"""Run both neural policies on recorded observations without a game engine."""
from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import sys


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"engine_bridge", "cg"}:
            raise ImportError("The replay demo must run without a native game engine")
        return None


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, help="JSONL of consecutive public observations for one player")
    parser.add_argument("--limit", type=int, default=8, help="0 processes all observations")
    parser.add_argument("--output", type=Path, help="Optional machine-readable predictions")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    root = args.package.resolve()
    source = args.input or root / "examples" / "observations.jsonl"
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoEngine())
    sys.path.insert(0, str(root))
    import torch
    from policy_bridge.decision_inference_bridge import PolicyRuntime, load_policy
    from policy_bridge.battlefield_environment.engine_choice_constraints import is_legal_action

    torch.set_num_threads(1)
    torch.manual_seed(42)
    deck = [int(value) for value in (root / "deck.csv").read_text().split()]
    policies = {}
    for label, filename in (("default", "candidate_policy.pt"), ("specialist", "opponent_routed_policy.pt")):
        model, _ = load_policy(root / filename, "cpu")
        policies[label] = PolicyRuntime(model, deck, device="cpu")
        policies[label].reset(deck)
    predictions = []
    with source.open() as handle, torch.inference_mode():
        for index, line in enumerate(handle):
            if args.limit and index >= args.limit:
                break
            row = json.loads(line)
            observation = row.get("observation", row)
            item = {"index": index, "turn": observation["current"]["turn"],
                    "option_count": len(observation["select"]["option"]), "policies": {}}
            for label, policy in policies.items():
                action, _, diagnostics = policy.decide(observation)
                legal = is_legal_action(action, observation["select"])
                if not legal:
                    raise RuntimeError(f"{label} returned an invalid action at observation {index}")
                item["policies"][label] = {"action": action, "legal": legal,
                    "log_probability": diagnostics["old_log_probability"],
                    "value": diagnostics["old_value"], "sanitized": diagnostics["sanitized"]}
            predictions.append(item)
            print(json.dumps(item), flush=True)
    result = {"status": "passed", "engine_used": False, "device": "cpu",
              "observations": len(predictions), "predictions": predictions,
              "scope": "Independent neural predictions on recorded states; no routing, rules overlay, search, or game transitions"}
    if not predictions:
        raise ValueError("No observations were processed")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}))


if __name__ == "__main__":
    run()
