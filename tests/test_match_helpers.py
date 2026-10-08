"""Unit checks of result interpretation and the random baseline's legal domain."""
import random
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_match import decks_for_seat, parse_args, random_action, read_deck, terminal_result


class ResultMappingTests(unittest.TestCase):
    def test_winner_is_absolute_seat_even_when_last_actor_differs(self):
        for winner in (0, 1):
            observation = {"current": {"result": winner, "yourIndex": 1 - winner},
                           "logs": [{"type": 23, "result": winner, "reason": 3}]}
            for candidate in (0, 1):
                result = terminal_result(observation, candidate)
                self.assertEqual(result["winner_seat"], winner)
                self.assertEqual(result["candidate_score"], float(candidate == winner))
                self.assertEqual(result["terminal_reason"], "no_active_pokemon")

    def test_draw_and_incomplete_are_distinct(self):
        for candidate in (0, 1):
            draw = terminal_result({"current": {"result": 2}}, candidate)
            ongoing = terminal_result({"current": {"result": -1}}, candidate)
            self.assertTrue(draw["terminal"])
            self.assertEqual(draw["candidate_score"], 0.5)
            self.assertIsNone(draw["winner_seat"])
            self.assertFalse(ongoing["terminal"])
            self.assertIsNone(ongoing["candidate_score"])

    def test_disagreement_and_unknown_code_fail(self):
        for observation in ({"current": {"result": 7}},
                            {"current": {"result": 0}, "logs": [{"type": 23, "result": 1}]},
                            {"current": {"result": -1}, "logs": [{"type": 23, "result": 0}]}):
            with self.assertRaises(ValueError):
                terminal_result(observation, 0)


class BaselineTests(unittest.TestCase):
    def test_default_arguments_keep_original_random_baseline(self):
        args = parse_args(["--package", "package", "--output", "output.json"])
        self.assertEqual(args.baseline_mode, "random")
        self.assertIsNone(args.baseline_deck)
        self.assertFalse(args.observations_all)
        self.assertFalse(args.seed_policy_runtime)

    def test_policy_and_all_observation_arguments(self):
        args = parse_args(["--package", "package", "--output", "output.json",
                           "--baseline-mode", "policy", "--baseline-deck", "other.csv",
                           "--observations-all", "--observation-limit", "1000"])
        self.assertEqual(args.baseline_mode, "policy")
        self.assertEqual(args.baseline_deck, "other.csv")
        self.assertTrue(args.observations_all)
        self.assertEqual(args.observation_limit, 1000)

    def test_two_distinct_decks_follow_candidate_seat(self):
        candidate, baseline = [7] * 60, [104] * 60
        self.assertEqual(decks_for_seat(candidate, baseline, 0), (candidate, baseline))
        self.assertEqual(decks_for_seat(candidate, baseline, 1), (baseline, candidate))

    def test_deck_reader_rejects_incomplete_decks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deck.csv"
            path.write_text("7\n" * 60)
            self.assertEqual(read_deck(path), [7] * 60)
            path.write_text("7\n" * 59)
            with self.assertRaises(ValueError):
                read_deck(path)

    def test_reproducible_legal_distinct_indices(self):
        select = {"minCount": 0, "maxCount": 5, "option": [{}, {}, {}, {}, {}]}
        left, right = random.Random(42), random.Random(42)
        actions = [random_action(select, left) for _ in range(150)]
        self.assertEqual(actions, [random_action(select, right) for _ in range(150)])
        self.assertEqual({len(action) for action in actions}, set(range(6)))
        for action in actions:
            self.assertEqual(len(action), len(set(action)))
            self.assertTrue(all(0 <= index < 5 for index in action))
        self.assertTrue(any(action != sorted(action) for action in actions if len(action) > 1))

    def test_forced_empty_and_clamped_maximum(self):
        rng = random.Random(13)
        self.assertEqual(random_action({"minCount": 0, "maxCount": 0, "option": []}, rng), [])
        action = random_action({"minCount": 3, "maxCount": 9, "option": [{}, {}, {}]}, rng)
        self.assertEqual(sorted(action), [0, 1, 2])

    def test_impossible_bounds_fail_instead_of_silent_repair(self):
        for minimum, maximum in ((2, 1), (4, 5), (-1, 2)):
            with self.assertRaises(ValueError):
                random_action({"minCount": minimum, "maxCount": maximum, "option": [{}, {}, {}]},
                              random.Random(1))


if __name__ == "__main__":
    unittest.main()
