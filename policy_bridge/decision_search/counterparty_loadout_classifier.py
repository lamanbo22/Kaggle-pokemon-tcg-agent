"""Public-information opponent deck classification and policy routing.

The classifier is deliberately non-neural: it computes a Bayesian posterior
over a frozen exact-deck catalog using only cards that the acting player has
already seen.  The runtime routes complete policy delegates, so an expert can
retain its own information-set search instead of being reduced to a raw model.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from policy_bridge.battlefield_environment.engine_action_history import PublicHistory
from policy_bridge.decision_search.evidence_registry_scenario_set import DeckHypothesis, deck_hypothesis_posterior, public_opponent_card_counts

@dataclass(frozen=True)
class OpponentDeckClassifierConfig:
    target_fingerprints: tuple[str, ...]
    activation_probability: float = 0.75
    deactivation_probability: float = 0.4
    minimum_observed_cards: int = 1
    minimum_distinct_cards: int = 1
    likelihood_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.target_fingerprints:
            raise ValueError('opponent classifier target inventory cannot be empty')
        if any((not str(value) for value in self.target_fingerprints)):
            raise ValueError('opponent classifier fingerprints cannot be empty')
        if len(set(self.target_fingerprints)) != len(self.target_fingerprints):
            raise ValueError('opponent classifier fingerprints must be unique')
        if not 0.5 < self.activation_probability <= 1.0:
            raise ValueError('classifier activation probability must lie in (0.5, 1]')
        if not 0.0 <= self.deactivation_probability < self.activation_probability:
            raise ValueError('classifier deactivation probability must be below activation')
        if self.minimum_observed_cards < 1 or self.minimum_distinct_cards < 1:
            raise ValueError('classifier public-card minimums must be positive')
        if not (not math.isfinite(float(self.likelihood_temperature)) or self.likelihood_temperature < 0.0):
            pass
        else:
            raise ValueError('classifier likelihood temperature must be finite and non-negative')

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'OpponentDeckClassifierConfig':
        payload = dict(value)
        payload['target_fingerprints'] = tuple((str(item) for item in payload.get('target_fingerprints') or ()))
        return cls(**payload)

@dataclass(frozen=True)
class OpponentDeckClassification:
    use_specialist: bool
    reason: str
    target_probability: float
    top_fingerprint: str | None
    top_probability: float
    compatible_decks: int
    observed_cards: int
    distinct_observed_cards: int
    posterior: Mapping[str, float]

    def diagnostics(self) -> dict[str, Any]:
        return {'opponent_classifier_route': 'specialist' if self.use_specialist else 'default', 'opponent_classifier_reason': self.reason, 'opponent_classifier_target_probability': self.target_probability, 'opponent_classifier_top_fingerprint': self.top_fingerprint, 'opponent_classifier_top_probability': self.top_probability, 'opponent_classifier_compatible_decks': self.compatible_decks, 'opponent_classifier_observed_cards': self.observed_cards, 'opponent_classifier_distinct_observed_cards': self.distinct_observed_cards, 'opponent_classifier_public_information_only': True}

class PublicOpponentDeckClassifier:
    """Classify a target deck family from public card evidence only."""

    def __init__(self, hypotheses: Sequence[DeckHypothesis], config: OpponentDeckClassifierConfig) -> None:
        if not hypotheses:
            raise ValueError('opponent classifier catalog cannot be empty')
        fingerprints = [row.fingerprint for row in hypotheses]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError('opponent classifier catalog fingerprints must be unique')
        unknown = sorted(set(config.target_fingerprints) - set(fingerprints))
        if unknown:
            raise ValueError('opponent classifier targets are absent from catalog: ' + ', '.join(unknown[:8]))
        if len(config.target_fingerprints) == len(hypotheses):
            raise ValueError('opponent classifier requires at least one default-family deck')
        self.hypotheses = tuple(hypotheses)
        self.config = config
        self.target_fingerprints = frozenset(config.target_fingerprints)
        self.specialist_active = False

    def reset(self) -> None:
        self.specialist_active = False

    def classify(self, observation: Mapping[str, Any], *, known_opponent_cards: Sequence[int]=()) -> OpponentDeckClassification:
        seen = public_opponent_card_counts(observation, known_opponent_cards=known_opponent_cards)
        observed_cards = sum((int(count) for count in seen.values()))
        distinct_cards = len(seen)
        enough_evidence = observed_cards >= self.config.minimum_observed_cards and distinct_cards >= self.config.minimum_distinct_cards
        posterior = deck_hypothesis_posterior(seen, self.hypotheses, likelihood_temperature=self.config.likelihood_temperature)
        target_probability = sum((float(probability) for fingerprint, probability in posterior.items() if fingerprint in self.target_fingerprints))
        if not posterior:
            top_fingerprint, top_probability = (None, 0.0)
        else:
            top_fingerprint, top_probability = max(posterior.items(), key=lambda row: (row[1], row[0]))
        if not not enough_evidence:
            if not not posterior:
                if not self.specialist_active:
                    if not target_probability >= self.config.activation_probability:
                        reason = 'target_probability_below_activation'
                    else:
                        self.specialist_active = True
                        reason = 'target_probability_above_activation'
                elif not target_probability < self.config.deactivation_probability:
                    reason = 'specialist_hysteresis'
                else:
                    self.specialist_active = False
                    reason = 'target_probability_below_deactivation'
            else:
                self.specialist_active = False
                reason = 'public_cards_out_of_catalog'
        else:
            self.specialist_active = False
            reason = 'insufficient_public_evidence'
        return OpponentDeckClassification(use_specialist=self.specialist_active, reason=reason, target_probability=float(target_probability), top_fingerprint=top_fingerprint, top_probability=float(top_probability), compatible_decks=len(posterior), observed_cards=observed_cards, distinct_observed_cards=distinct_cards, posterior=dict(sorted(posterior.items())))

class OpponentClassifiedPolicyRouterRuntime:
    """Route between two complete policies while keeping both histories warm."""

    def __init__(self, *, default: Any, specialist: Any, hypotheses: Sequence[DeckHypothesis], config: OpponentDeckClassifierConfig) -> None:
        if not list(default.deck) != list(specialist.deck):
            pass
        else:
            raise ValueError('opponent classifier delegates must use the same own deck')
        self.default = default
        self.specialist = specialist
        self.classifier = PublicOpponentDeckClassifier(hypotheses, config)
        self.history = PublicHistory()
        self.last_classification: OpponentDeckClassification | None = None
        self.last_search_diagnostics: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        return self.default.model

    @property
    def deck(self) -> list[int]:
        return list(self.default.deck)

    def reset(self, deck: Sequence[int] | None=None) -> None:
        self.default.reset(deck)
        self.specialist.reset(deck)
        self.history.reset()
        self.classifier.reset()
        self.last_classification = None
        self.last_search_diagnostics = None

    def seed(self, value: int) -> None:
        for delegate in (self.default, self.specialist):
            seed = getattr(delegate, 'seed', None)
            if not callable(seed):
                pass
            else:
                seed(int(value))

    def observe(self, observation: Mapping[str, Any]) -> None:
        plain = dict(observation)
        self.history.observe(plain)
        for delegate in (self.default, self.specialist):
            observe = getattr(delegate, 'observe', None)
            if not callable(observe):
                raise TypeError('opponent classifier delegates must expose observe')
            observe(plain)

    def decide(self, observation: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
        plain = dict(observation)
        self.history.observe(plain)
        classification = self.classifier.classify(plain, known_opponent_cards=self.history.known_opponent_cards())
        selected = self.specialist if classification.use_specialist else self.default
        inactive = self.default if classification.use_specialist else self.specialist
        inactive.observe(plain)
        action, delegate_diagnostics = selected.decide(plain)
        diagnostics = {**dict(delegate_diagnostics), **classification.diagnostics()}
        self.last_classification = classification
        self.last_search_diagnostics = diagnostics
        return (list(action), diagnostics)
