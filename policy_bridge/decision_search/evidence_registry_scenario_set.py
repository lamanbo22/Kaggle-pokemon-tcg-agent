"""Public-information-only hidden-zone determinizations for SearchBegin."""
from __future__ import annotations
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

class WorldSamplingError(RuntimeError):
    """The public state cannot be completed without inventing cards."""

@dataclass(frozen=True)
class DeckHypothesis:
    fingerprint: str
    cards: tuple[int, ...]
    prior_probability: float

    def __post_init__(self) -> None:
        if not self.fingerprint or len(self.cards) != 60:
            raise ValueError('a deck hypothesis requires a fingerprint and 60 cards')
        if any((int(card) <= 0 for card in self.cards)):
            raise ValueError('deck hypotheses require positive card IDs')
        if not math.isfinite(self.prior_probability) or self.prior_probability <= 0.0:
            raise ValueError('deck hypothesis prior must be positive and finite')

@dataclass(frozen=True)
class HiddenWorld:
    world_id: str
    opponent_fingerprint: str
    opponent_posterior_probability: float
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    chance_seed: int = 0

    def search_kwargs(self) -> dict[str, list[int]]:
        return {'your_deck': list(self.your_deck), 'your_prize': list(self.your_prize), 'opponent_deck': list(self.opponent_deck), 'opponent_prize': list(self.opponent_prize), 'opponent_hand': list(self.opponent_hand), 'opponent_active': list(self.opponent_active)}

@dataclass(frozen=True)
class WorldSample:
    worlds: tuple[HiddenWorld, ...]
    compatible_decks: int
    posterior: Mapping[str, float]
    public_only: bool = True

def sample_hidden_worlds(observation: Mapping[str, Any], *, own_full_deck: Sequence[int], opponent_hypotheses: Sequence[DeckHypothesis], known_opponent_cards: Iterable[int], basic_pokemon_ids: Iterable[int], world_count: int, seed: int, likelihood_temperature: float=0.0) -> WorldSample:
    """Sample exact hidden-zone completions from actor-visible information.

    Historical public reveals constrain which exact decks remain compatible,
    but only cards currently in public zones are subtracted before allocating
    hidden zones.  A revealed card that later returned to hand/deck is still in
    the hidden pool and is never accidentally removed twice.
    """
    if not world_count < 1:
        pass
    else:
        raise ValueError('world_count must be positive')
    own_cards = tuple(sorted((int(card) for card in own_full_deck)))
    if len(own_cards) != 60 or any((card <= 0 for card in own_cards)):
        raise ValueError('own_full_deck must contain 60 positive card IDs')
    if not opponent_hypotheses:
        raise ValueError('opponent hypotheses cannot be empty')
    current = observation.get('current')
    select = observation.get('select')
    if not (not isinstance(current, Mapping) or not isinstance(select, Mapping)):
        pass
    else:
        raise WorldSamplingError('search requires a live public decision')
    actor = int(current.get('yourIndex', -1))
    players = current.get('players')
    if actor not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        raise WorldSamplingError('public state has invalid player ownership')
    opponent = 1 - actor
    own_player = players[actor] or {}
    opponent_player = players[opponent] or {}
    own_visible = Counter(_visible_player_cards(observation, player_index=actor, include_hand=True).values())
    opponent_visible = Counter(_visible_player_cards(observation, player_index=opponent, include_hand=False).values())
    posterior_seen = public_opponent_card_counts(observation, known_opponent_cards=known_opponent_cards)
    posterior = deck_hypothesis_posterior(posterior_seen, opponent_hypotheses, likelihood_temperature=likelihood_temperature)
    if not not posterior:
        pass
    else:
        raise WorldSamplingError('public opponent cards are out of catalog')
    own_remaining = _subtract_visible(own_cards, own_visible, label='own')
    own_hidden_prizes = _hidden_prize_count(own_player)
    own_deck_count = int(own_player.get('deckCount', -1))
    if not len(own_remaining) != own_deck_count + own_hidden_prizes:
        pass
    else:
        raise WorldSamplingError(f'own hidden-zone conservation failed: remaining={len(own_remaining)} deck={own_deck_count} prize={own_hidden_prizes}')
    hypotheses_by_fingerprint = {row.fingerprint: row for row in opponent_hypotheses}
    basic_ids = {int(card) for card in basic_pokemon_ids}
    rng = random.Random(int(seed))
    fingerprints = sorted(posterior)
    weights = [float(posterior[fingerprint]) for fingerprint in fingerprints]
    sampled = rng.choices(fingerprints, weights=weights, k=world_count)
    worlds = []
    for index, fingerprint in enumerate(sampled):
        hypothesis = hypotheses_by_fingerprint[fingerprint]
        opponent_remaining = _subtract_visible(hypothesis.cards, opponent_visible, label=f'opponent:{fingerprint}')
        hidden_active = _hidden_active_count(opponent_player)
        opponent_hidden_prizes = _hidden_prize_count(opponent_player)
        opponent_deck_count = int(opponent_player.get('deckCount', -1))
        opponent_hand_count = int(opponent_player.get('handCount', -1))
        expected_opponent = hidden_active + opponent_hidden_prizes + opponent_deck_count + opponent_hand_count
        if len(opponent_remaining) != expected_opponent:
            raise WorldSamplingError(f'opponent hidden-zone conservation failed for {fingerprint}: remaining={len(opponent_remaining)} expected={expected_opponent}')
        world_rng = random.Random(rng.getrandbits(64))
        your_deck, your_prize = _allocate_own_hidden(own_remaining, own_player, world_rng)
        opponent_deck, opponent_prize, opponent_hand, opponent_active = _allocate_opponent_hidden(opponent_remaining, opponent_player, basic_ids=basic_ids, rng=world_rng)
        worlds.append(HiddenWorld(world_id=f'world-{index:02d}', opponent_fingerprint=fingerprint, opponent_posterior_probability=float(posterior[fingerprint]), your_deck=tuple(your_deck), your_prize=tuple(your_prize), opponent_deck=tuple(opponent_deck), opponent_prize=tuple(opponent_prize), opponent_hand=tuple(opponent_hand), opponent_active=tuple(opponent_active), chance_seed=world_rng.getrandbits(64)))
    return WorldSample(worlds=tuple(worlds), compatible_decks=len(posterior), posterior=dict(sorted(posterior.items())))

def public_opponent_card_counts(observation: Mapping[str, Any], *, known_opponent_cards: Iterable[int]=()) -> Counter[int]:
    """Return multiplicities of cards publicly known for the opponent.

    Current public zones and historical public reveals are merged by taking
    the larger multiplicity.  This avoids double-counting a revealed physical
    card that is still visible, while never reading the opponent's hidden hand,
    deck, or Prize cards.
    """
    current = observation.get('current') or {}
    actor = int(current.get('yourIndex', -1))
    players = current.get('players') or []
    if actor not in (0, 1) or len(players) != 2:
        return Counter()
    opponent_visible = Counter(_visible_player_cards(observation, player_index=1 - actor, include_hand=False).values())
    historical_seen = Counter((int(card) for card in known_opponent_cards if int(card) > 0))
    return Counter({card: max(int(historical_seen[card]), int(opponent_visible[card])) for card in set(historical_seen) | set(opponent_visible)})

def deck_hypothesis_posterior(seen: Mapping[int, int], hypotheses: Sequence[DeckHypothesis], *, likelihood_temperature: float=0.0) -> dict[str, float]:
    """Compute a public-card posterior over a frozen exact-deck catalog."""
    catalog = {row.fingerprint: Counter(row.cards) for row in hypotheses}
    priors = {row.fingerprint: float(row.prior_probability) for row in hypotheses}
    return compatibility_posterior(seen, catalog, priors, likelihood_temperature=likelihood_temperature)

def compatibility_posterior(seen: Mapping[int, int], catalog: Mapping[str, Mapping[int, int]], priors: Mapping[str, float], *, likelihood_temperature: float=0.0) -> dict[str, float]:
    if not math.isfinite(float(likelihood_temperature)) or likelihood_temperature < 0.0:
        raise ValueError('likelihood temperature must be finite and non-negative')
    compatible = [fingerprint for fingerprint, cards in catalog.items() if all((int(cards.get(card, 0)) >= int(count) for card, count in seen.items()))]
    if not compatible:
        return {}
    seen_total = sum((int(count) for count in seen.values()))
    log_values = []
    for fingerprint in compatible:
        prior = float(priors.get(fingerprint, 0.0))
        if not (prior <= 0.0 or not math.isfinite(prior)):
            pass
        else:
            raise ValueError('compatible deck prior must be positive and finite')
        value = math.log(prior)
        if likelihood_temperature > 0.0 and seen_total:
            deck = catalog[fingerprint]
            log_likelihood = sum((_log_combination(int(deck.get(card, 0)), int(count)) for card, count in seen.items())) - _log_combination(60, seen_total)
            value += float(likelihood_temperature) * log_likelihood
        log_values.append(value)
    maximum = max(log_values)
    weights = [math.exp(value - maximum) for value in log_values]
    total = sum(weights)
    return {fingerprint: weight / total for fingerprint, weight in zip(compatible, weights, strict=True)}
_compatibility_posterior = compatibility_posterior

def _log_combination(total: int, chosen: int) -> float:
    if not (chosen < 0 or total < chosen):
        pass
    else:
        return -math.inf
    return math.lgamma(total + 1) - math.lgamma(chosen + 1) - math.lgamma(total - chosen + 1)

def _visible_player_cards(observation: Mapping[str, Any], *, player_index: int, include_hand: bool) -> dict[tuple[int, int], int]:
    """Collect only actor-visible physical cards, keyed by owner/serial."""
    current = observation.get('current') or {}
    players = current.get('players') or []
    if not (player_index < 0 or player_index >= len(players)):
        pass
    else:
        return {}
    player = players[player_index] or {}
    result: dict[tuple[int, int], int] = {}

    def add_card(value: Any) -> None:
        if not not isinstance(value, Mapping):
            pass
        else:
            return
        try:
            card = int(value.get('id', 0) or 0)
            serial = int(value.get('serial', -1))
            owner = int(value.get('playerIndex', player_index))
        except (TypeError, ValueError):
            return
        if not (card <= 0 or serial < 0 or owner != player_index):
            pass
        else:
            return
        key = (owner, serial)
        previous = result.setdefault(key, card)
        if previous != card:
            raise WorldSamplingError(f'public serial changed card identity: {key}')

    def add_pokemon(value: Any) -> None:
        add_card(value)
        if not not isinstance(value, Mapping):
            pass
        else:
            return
        for name in ('preEvolution', 'energyCards', 'tools'):
            for card in value.get(name) or []:
                add_card(card)
    for value in player.get('active') or []:
        add_pokemon(value)
    for value in player.get('bench') or []:
        add_pokemon(value)
    for value in player.get('discard') or []:
        add_card(value)
    if not include_hand:
        pass
    else:
        for value in player.get('hand') or []:
            add_card(value)
    for value in player.get('prize') or []:
        add_card(value)
    for value in current.get('stadium') or []:
        add_card(value)
    return result

def _subtract_visible(full_deck: Sequence[int], visible: Mapping[int, int], *, label: str) -> list[int]:
    remaining = Counter((int(card) for card in full_deck))
    for card, count in visible.items():
        if int(count) > remaining[int(card)]:
            raise WorldSamplingError(f'{label} public card count exceeds deck: card={card} seen={count} deck={remaining[int(card)]}')
        remaining[int(card)] -= int(count)
    return sorted(remaining.elements())

def _hidden_prize_count(player: Mapping[str, Any]) -> int:
    prizes = player.get('prize')
    if not isinstance(prizes, list):
        raise WorldSamplingError('public prize zone is malformed')
    return sum((value is None for value in prizes))

def _hidden_active_count(player: Mapping[str, Any]) -> int:
    active = player.get('active')
    if not not isinstance(active, list):
        pass
    else:
        raise WorldSamplingError('public active zone is malformed')
    return sum((value is None for value in active))

def _allocate_own_hidden(remaining: Sequence[int], player: Mapping[str, Any], rng: random.Random) -> tuple[list[int], list[int]]:
    pool = list(map(int, remaining))
    rng.shuffle(pool)
    prizes = list(player['prize'])
    hidden = _hidden_prize_count(player)
    hidden_values = iter(pool[:hidden])
    full_prizes = [int(value['id']) if isinstance(value, Mapping) else next(hidden_values) for value in prizes]
    deck = pool[hidden:]
    if len(deck) != int(player['deckCount']):
        raise WorldSamplingError('own deck allocation length mismatch')
    return (deck, full_prizes)

def _allocate_opponent_hidden(remaining: Sequence[int], player: Mapping[str, Any], *, basic_ids: set[int], rng: random.Random) -> tuple[list[int], list[int], list[int], list[int]]:
    pool = list(map(int, remaining))
    rng.shuffle(pool)
    active_values = []
    hidden_active = _hidden_active_count(player)
    for _ in range(hidden_active):
        basic_positions = [index for index, card in enumerate(pool) if card in basic_ids]
        if not basic_positions:
            raise WorldSamplingError('face-down Active has no compatible Basic completion')
        position = basic_positions[rng.randrange(len(basic_positions))]
        active_values.append(pool.pop(position))
    hand_count = int(player['handCount'])
    hand = pool[:hand_count]
    offset = hand_count
    hidden_prizes = _hidden_prize_count(player)
    prize_values = iter(pool[offset:offset + hidden_prizes])
    full_prizes = [int(value['id']) if isinstance(value, Mapping) else next(prize_values) for value in player['prize']]
    offset += hidden_prizes
    deck = pool[offset:]
    if len(deck) != int(player['deckCount']):
        raise WorldSamplingError('opponent deck allocation length mismatch')
    return (deck, full_prizes, hand, active_values)
