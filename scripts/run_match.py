"""Run one CPU match against a random or independent base-policy baseline.

The referee and candidate use the submitted native library. Each agent receives
its acting-player observations, never visualizer output. Launch in a fresh process for
each match. A parent process must impose a hard subprocess timeout because a
cooperative Python wall limit cannot interrupt a stuck native-library call.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from typing import Any


BASELINE_DESCRIPTION = (
    "At each decision choose the selection count "
    "uniformly from [minCount, min(maxCount, len(select.option))], then use "
    "random.Random(seed).sample to choose that many distinct option indices. "
    "Preserve sampled order, including ordered selections. This is uniform over "
    "counts and ordered selections conditional on count, not over all action sets."
)
POLICY_BASELINE_DESCRIPTION = (
    "Independent PolicyRuntime using candidate.DEFAULT_MODEL on CPU, the baseline "
    "deck, its own history and the submitted RULE_PROFILE. Greedy temperature=0; "
    "no opponent classifier or search. The shared model is used for inference only."
)


def read_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError(f"{path} must contain exactly 60 positive card IDs")
    return cards


def decks_for_seat(candidate: list[int], baseline: list[int], seat: int) -> tuple[list[int], list[int]]:
    if seat not in (0, 1):
        raise ValueError("candidate seat must be 0 or 1")
    return (candidate, baseline) if seat == 0 else (baseline, candidate)


def random_action(select: dict[str, Any], rng: random.Random) -> list[int]:
    """Sample only the engine-provided choices within its cardinality bounds."""
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0))
    maximum = min(int(select.get("maxCount", len(options))), len(options))
    if not 0 <= minimum <= maximum:
        raise ValueError(f"invalid selection bounds: {minimum=}, {maximum=}")
    return rng.sample(range(len(options)), rng.randint(minimum, maximum))


def terminal_result(observation: dict[str, Any], candidate_seat: int) -> dict[str, Any]:
    """Use absolute engine winner indices, never the final acting-player seat."""
    if candidate_seat not in (0, 1):
        raise ValueError("candidate_seat must be 0 or 1")
    result = int((observation.get("current") or {}).get("result", -1))
    if result not in (-1, 0, 1, 2):
        raise ValueError(f"unknown engine result: {result}")
    log = next((row for row in reversed(observation.get("logs") or [])
                if int(row.get("type", -1)) == 23), None)
    if log is not None:
        if result == -1:
            raise ValueError("nonterminal engine state contains a RESULT log")
        if log.get("result") is not None and int(log["result"]) != result:
            raise ValueError("current.result and terminal RESULT log disagree")
    winner = result if result in (0, 1) else None
    score = None if result == -1 else 0.5 if result == 2 else float(winner == candidate_seat)
    reason = log.get("reason") if log else None
    labels = {1: "prizes_exhausted", 2: "deck_empty_at_turn_start",
              3: "no_active_pokemon", 4: "card_effect"}
    return {
        "terminal": result != -1,
        "engine_result": result,
        "winner_seat": winner,
        "candidate_score": score,
        "candidate_outcome": "incomplete" if result == -1 else "draw" if result == 2
                             else "win" if winner == candidate_seat else "loss",
        "terminal_reason_code": reason,
        "terminal_reason": labels.get(reason, "missing" if reason is None else f"unknown_{reason}"),
        "result_log_found": log is not None,
    }


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "max_ms": max(values) if values else None}


def enum_label(enum: Any, value: Any) -> str:
    try:
        return enum(int(value)).name
    except (ValueError, TypeError):
        return str(value)


def output_path(value: str | None, package: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).resolve()
    if path == package or package in path.parents:
        raise ValueError("output paths must be outside the submission package")
    if path.exists():
        raise FileExistsError(f"Use a new output path; existing evidence is preserved: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    start = time.monotonic()
    package = Path(args.package).resolve()
    output = output_path(args.output, package)
    replay = output_path(args.replay, package)
    trace_path = output_path(args.trace, package)
    observations_path = output_path(args.observations, package)
    counters: Counter[str] = Counter()
    selection_types = {"candidate": Counter(), "baseline": Counter()}
    selection_contexts = {"candidate": Counter(), "baseline": Counter()}
    option_types = {"candidate": Counter(), "baseline": Counter()}
    search_reasons: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    candidate_latencies: list[float] = []
    baseline_latencies: list[float] = []
    search_latencies: list[float] = []
    search_events: list[dict[str, Any]] = []
    action_digest = hashlib.sha256()
    trace_handle = None
    observations_handle = None
    observed_contexts: set[tuple[Any, Any]] = set()
    saved_observations = 0
    battle_active = False
    game = None
    observation: dict[str, Any] = {}
    stage = "setup"
    last_action = None
    last_diagnostics = None
    baseline_runtime = None
    random_device = None
    max_turn = -1
    summary: dict[str, Any] = {
        "schema": "pokemon-submission-local-match-v1",
        "status": "starting", "package": str(package), "candidate_seat": args.seat,
        "seed": args.seed, "baseline_seed": args.seed, "engine_seed": args.engine_seed,
        "baseline": BASELINE_DESCRIPTION if args.baseline_mode == "random" else POLICY_BASELINE_DESCRIPTION,
        "baseline_mode": args.baseline_mode,
        "limits": {"max_decisions": args.max_decisions, "wall_seconds": args.wall_seconds,
                   "wall_limit_kind": "cooperative; caller must enforce a hard subprocess timeout"},
        "training_performed": False, "device": "cpu",
        "reproducibility": {"python_numpy_torch_seeded": False, "baseline_seeded": True,
                            "engine_rng_controlled": False,
                            "full_match_determinism_verified": False,
                            "note": "Search deadlines may also affect repeatability."},
    }
    exit_code = 1
    try:
        # These settings precede every submission, numpy and torch import.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(package))
        random.seed(args.seed)
        baseline_rng = random.Random(args.seed)
        emit("loading", package=str(package), seat=args.seat, seed=args.seed)
        import numpy as np
        import torch
        np.random.seed(args.seed % (2 ** 32))
        torch.manual_seed(args.seed)
        summary["reproducibility"]["python_numpy_torch_seeded"] = True
        spec = importlib.util.spec_from_file_location("_pokemon_validation_candidate", package / "main.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot import submission main.py")
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        from engine_bridge import engine_battle_state as game
        from engine_bridge import engine_action_gateway as api
        from policy_bridge.battlefield_environment.engine_choice_constraints import is_legal_action

        devices = {str(parameter.device) for model in (candidate.DEFAULT_MODEL, candidate.SPECIALIST_MODEL)
                   for parameter in model.parameters()}
        if devices != {"cpu"}:
            raise RuntimeError(f"models not exclusively on CPU: {devices}")
        summary["model_devices"] = sorted(devices)
        summary["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                               "numpy": np.__version__}
        summary["inputs_sha256"] = {
            name: sha256_file(package / name) for name in (
                "main.py", "deck.csv", "candidate_policy.pt", "opponent_routed_policy.pt",
                "engine_bridge/battle_runtime_bridge.so")}
        deck = read_deck(package / "deck.csv")
        baseline_deck_path = Path(args.baseline_deck).resolve() if args.baseline_deck else package / "deck.csv"
        baseline_deck = read_deck(baseline_deck_path)
        summary["baseline_deck"] = {"path": str(baseline_deck_path),
                                    "sha256": sha256_file(baseline_deck_path),
                                    "same_cards_as_candidate": sorted(baseline_deck) == sorted(deck),
                                    "cards": baseline_deck}
        if args.baseline_mode == "policy":
            from policy_bridge.decision_inference_bridge import PolicyRuntime
            baseline_runtime = PolicyRuntime(candidate.DEFAULT_MODEL, baseline_deck,
                                             device="cpu", rule_profile=candidate.RULE_PROFILE)
            baseline_runtime.reset(baseline_deck)
            summary["baseline_policy"] = {"checkpoint": "candidate_policy.pt",
                                           "independent_history": True,
                                           "temperature": 0, "routing": False, "search": False,
                                           "rule_profile": candidate.RULE_PROFILE}
        stage = "candidate_reset"
        initial = {"current": None, "logs": [], "remainingOverageTime": 600,
                   "search_begin_input": None, "select": None, "step": 0}
        returned_deck = candidate.agent(initial)
        if returned_deck != deck:
            raise ValueError("candidate initial deck differs from submitted deck.csv")
        # Exercise the deployed reset entrypoint. Keep its fixed search seed unless
        # the caller explicitly requests the package's optional episode seed API.
        summary["candidate_reset"] = "main.agent(select=None)"
        if args.seed_policy_runtime:
            candidate._POLICY.seed(args.seed)
        summary["candidate_runtime_seed_override"] = bool(args.seed_policy_runtime)
        summary["candidate_runtime_config"] = candidate.RUNTIME_CONFIG
        summary["classifier_config"] = candidate.CLASSIFIER_CONFIG
        stage = "engine_start"
        if args.engine_seed is not None:
            from policy_bridge.battlefield_environment.repeatable_engine_rng import DeterministicRandomDevice
            random_device = DeterministicRandomDevice()
            random_device.seed(args.engine_seed)
            summary["reproducibility"].update(
                engine_rng_controlled=True, engine_rng_shim=str(random_device.path),
                engine_rng_shim_sha256=sha256_file(random_device.path))
        elif os.environ.get("PTCG_RANDOM_DEVICE_SHIM"):
            from policy_bridge.battlefield_environment.repeatable_engine_rng import DeterministicRandomDevice
            random_device = DeterministicRandomDevice()
            random_device.disable()
        observation, start_data = game.battle_start(*decks_for_seat(deck, baseline_deck, args.seat))
        battle_active = bool(start_data.battlePtr)
        summary["battle_start"] = {"error_player": start_data.errorPlayer, "error_type": start_data.errorType}
        if observation is None:
            raise RuntimeError(f"BattleStart rejected decks: {summary['battle_start']}")
        summary["initial_observation_sha256"] = fingerprint(observation)
        summary["first_player"] = observation["current"].get("firstPlayer")
        if trace_path:
            trace_handle = trace_path.open("w", encoding="utf-8")
        if observations_path:
            observations_handle = observations_path.open("w", encoding="utf-8")
        emit("started", first_player=summary["first_player"],
             engine_rng_controlled=summary["reproducibility"]["engine_rng_controlled"],
             elapsed_seconds=round(time.monotonic() - start, 3))
        with torch.inference_mode():
            while True:
                stage = "terminal_check"
                terminal = terminal_result(observation, args.seat)
                current = observation.get("current") or {}
                max_turn = max(max_turn, int(current.get("turn", -1)))
                if terminal["terminal"]:
                    summary.update(terminal, status="completed")
                    summary["terminal_observation"] = observation
                    exit_code = 0
                    break
                if counters["accepted_decisions"] >= args.max_decisions:
                    summary.update(terminal, status="decision_limit")
                    break
                if time.monotonic() - start >= args.wall_seconds:
                    summary.update(terminal, status="wall_limit")
                    break
                seat = int(current["yourIndex"])
                if seat not in (0, 1):
                    raise ValueError(f"invalid acting seat {seat}")
                role = "candidate" if seat == args.seat else "baseline"
                select = observation.get("select")
                if not isinstance(select, dict):
                    raise ValueError("nonterminal observation has no selection")
                stage = f"{role}_action"
                counters[f"{role}_attempts"] += 1
                observation_hash = fingerprint(observation)
                context_key = (select.get("type"), select.get("context"))
                save_observation = (observations_handle is not None and role == "candidate"
                                    and saved_observations < args.observation_limit
                                    and (args.observations_all or saved_observations < 5
                                         or context_key not in observed_contexts))
                # Serialize before decide so this evidence is independent of any
                # in-place mutation an agent might make to its input dictionary.
                observation_json = json.dumps(observation) if save_observation else None
                action_start = time.monotonic()
                if role == "candidate":
                    action = candidate.agent(observation)
                elif baseline_runtime is not None:
                    action, _, _ = baseline_runtime.decide(observation, temperature=0.0)
                else:
                    action = random_action(select, baseline_rng)
                action_ms = (time.monotonic() - action_start) * 1000
                if role == "baseline":
                    baseline_latencies.append(action_ms)
                last_action = {"seat": seat, "role": role, "action": action,
                               "turn": current.get("turn"), "select_type": select.get("type"),
                               "select_context": select.get("context"), "option_count": len(select.get("option") or []),
                               "min_count": select.get("minCount"), "max_count": select.get("maxCount")}
                if role == "candidate":
                    candidate_latencies.append(action_ms)
                    last_diagnostics = candidate._POLICY.last_search_diagnostics
                    if isinstance(last_diagnostics, dict):
                        counters["candidate_diagnostics"] += 1
                        search_reasons[str(last_diagnostics.get("search_reason"))] += 1
                        routes[str(last_diagnostics.get("opponent_classifier_route"))] += 1
                        for key in ("search_attempted", "search_applied"):
                            counters[key] += int(bool(last_diagnostics.get(key)))
                        if last_diagnostics.get("search_attempted"):
                            search_events.append({"decision": counters["accepted_decisions"],
                                                  "turn": current.get("turn"), **last_diagnostics})
                            if "elapsed_ms" in last_diagnostics:
                                search_latencies.append(float(last_diagnostics["elapsed_ms"]))
                if not is_legal_action(action, select):
                    counters[f"{role}_invalid_actions"] += 1
                    raise ValueError(f"{role} returned invalid option indices/cardinality: {last_action}")
                row = {"decision": counters["accepted_decisions"], **last_action,
                       "observation_sha256": observation_hash, "latency_ms": action_ms}
                if save_observation:
                    observations_handle.write(json.dumps({
                        **row, "observation": json.loads(observation_json),
                        "diagnostics": last_diagnostics,
                        "scope": "all candidate decisions up to cap" if args.observations_all else
                                 "first five candidate decisions, then first new select type/context",
                    }, sort_keys=True) + "\n")
                    observations_handle.flush()
                    saved_observations += 1
                    observed_contexts.add(context_key)
                action_digest.update(json.dumps({key: value for key, value in row.items() if key != "latency_ms"},
                                                sort_keys=True, separators=(",", ":")).encode() + b"\n")
                stage = f"engine_select_{role}"
                observation = game.battle_select(action)
                counters["accepted_decisions"] += 1
                counters[f"{role}_accepted_decisions"] += 1
                selection_types[role][enum_label(api.SelectType, select.get("type"))] += 1
                selection_contexts[role][enum_label(api.SelectContext, select.get("context"))] += 1
                for index in action:
                    option_types[role][enum_label(api.OptionType, select["option"][index].get("type"))] += 1
                if trace_handle:
                    trace_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    trace_handle.flush()
                if counters["accepted_decisions"] % args.progress_every == 0:
                    emit("progress", decisions=counters["accepted_decisions"], turn=max_turn,
                         candidate_decisions=counters["candidate_accepted_decisions"],
                         search_attempted=counters["search_attempted"], search_applied=counters["search_applied"],
                         elapsed_seconds=round(time.monotonic() - start, 3))
    except Exception as exc:
        summary.update(status="error", error_stage=stage, error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc(), last_action=last_action)
    finally:
        if battle_active and game is not None:
            if replay:
                try:
                    replay.write_text(game.visualize_data(), encoding="utf-8")
                    summary["visualizer_replay"] = str(replay)
                except Exception as exc:
                    summary["replay_error"] = repr(exc)
            try:
                game.battle_finish()
                summary["battle_finished"] = True
            except Exception as exc:
                summary["cleanup_error"] = repr(exc)
                exit_code = 1
        if trace_handle:
            trace_handle.close()
        if observations_handle:
            observations_handle.close()
        if trace_path:
            summary["action_trace"] = str(trace_path)
        if observations_path:
            summary["candidate_observations"] = str(observations_path)
            summary["saved_candidate_observations"] = saved_observations
            summary["candidate_observations_complete"] = bool(
                args.observations_all and saved_observations == counters["candidate_accepted_decisions"]
                and summary["status"] == "completed")
        if random_device is not None:
            summary["native_random_device_calls"] = random_device.calls
        current = (observation.get("current") or {}) if observation else {}
        players = current.get("players") or []
        summary.update(
            elapsed_seconds=time.monotonic() - start, max_turn=max_turn,
            counters=dict(counters),
            selection_types={role: dict(counts) for role, counts in selection_types.items()},
            selection_contexts={role: dict(counts) for role, counts in selection_contexts.items()},
            selected_option_types={role: dict(counts) for role, counts in option_types.items()},
            candidate_action_latency=latency_summary(candidate_latencies),
            baseline_action_latency=latency_summary(baseline_latencies),
            search_latency=latency_summary(search_latencies), search_reasons=dict(search_reasons),
            classifier_routes=dict(routes), search_events=search_events,
            last_candidate_diagnostics=last_diagnostics,
            action_state_trace_sha256=action_digest.hexdigest(),
            final_players=[{"seat": index, "deck_count": player.get("deckCount"),
                            "prizes_remaining": len(player.get("prize") or [])}
                           for index, player in enumerate(players)],
        )
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit("final", **summary)
    return summary, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", "--baseline-seed", dest="seed", type=int, default=42,
                        help="Python, NumPy, Torch and baseline seed; packaged search seed stays unchanged")
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--baseline-deck", help="Optional 60-line card-ID deck; defaults to package/deck.csv")
    parser.add_argument("--baseline-mode", choices=("random", "policy"), default="random")
    parser.add_argument("--engine-seed", type=int, default=None,
                        help="Requires PTCG_RANDOM_DEVICE_SHIM preloaded before Python starts")
    parser.add_argument("--seed-policy-runtime", action="store_true",
                        help="Override packaged search episode seed via its optional seed API")
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--trace", help="Optional JSONL of accepted actions and observation hashes")
    parser.add_argument("--observations", help="Optional JSONL of representative full candidate inputs and actions")
    parser.add_argument("--observation-limit", type=int, default=64)
    parser.add_argument("--observations-all", action="store_true",
                        help="Save every candidate observation up to --observation-limit (use 1000 for full sequences)")
    parser.add_argument("--replay", help="Optional native visualizer output; never supplied to agents")
    args = parser.parse_args(argv)
    if args.max_decisions <= 0 or args.wall_seconds <= 0 or args.progress_every <= 0 or args.observation_limit <= 0:
        parser.error("limits and progress interval must be positive")
    return args


if __name__ == "__main__":
    _, code = run(parse_args())
    raise SystemExit(code)
