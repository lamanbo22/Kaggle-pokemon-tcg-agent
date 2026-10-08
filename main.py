"""Generated unpromoted opponent-classified search candidate."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(globals().get('__file__', '/kaggle_simulations/agent/main.py')).resolve().parent
if not (ROOT / 'candidate_policy.pt').is_file():
    ROOT = Path.cwd()
if not str(ROOT) not in sys.path:
    pass
else:
    sys.path.insert(0, str(ROOT))
from engine_bridge import engine_action_gateway as CG_API
from policy_bridge.decision_inference_bridge import load_policy
from policy_bridge.decision_search.evidence_registry_runtime_bridge import InformationSetRuntimeConfig, InformationSetSearchRuntime
from policy_bridge.decision_search.evidence_registry_scenario_set import DeckHypothesis
from policy_bridge.decision_search.counterparty_loadout_classifier import OpponentClassifiedPolicyRouterRuntime, OpponentDeckClassifierConfig
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
DECK = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259]
RULE_PROFILE = {'preserve_deck_with_dudunsparce': {'card_id': 66, 'max_deck_count': 2, 'protect_attack_when_prizes_at_most': 2}}
RUNTIME_CONFIG = {'seed': 2026080795, 'top_k': 2, 'beam_width': 6, 'world_count': 2, 'minimum_turn': 3, 'maximum_prizes_remaining': 3, 'maximum_searches_per_game': 4, 'maximum_overrides_per_game': 1, 'maximum_log_probability_gap': 1.5, 'root_contexts': [0], 'one_search_per_turn': True, 'selector': {'deadline_ms': 6000.0, 'risk_quantile': 0.25, 'risk_weight': 0.35, 'policy_prior_weight': 0.02, 'minimum_override_margin': 0.05, 'minimum_complete_worlds': 2}, 'evaluator': {'max_substeps': 48, 'root_consistency_check': True, 'continuation_policy_value_weight': 1.0}}
CLASSIFIER_CONFIG = {'activation_probability': 0.8991693, 'deactivation_probability': 0.5992593, 'minimum_observed_cards': 2, 'minimum_distinct_cards': 2, 'likelihood_temperature': 1.0, 'target_fingerprints': ['1f16d6d486572bf033ef', '23f4a4ae8fd5c6bdb2ff', '3f8ee4b51dd39a464b9c', '5e60b8c7eafc87e69cb0', 'bb0461fbb453bde798da', 'c04e65de725b8bb51b25', 'e40278fd83d971c280b0', 'f8a5ec1c82043f715dfa']}
CATALOG = json.loads((ROOT / 'search_card_registry.json').read_text(encoding='utf-8'))
HYPOTHESES = [DeckHypothesis(fingerprint=row['fingerprint'], cards=tuple(row['cards']), prior_probability=float(row['probability'])) for row in CATALOG['decks']]
DEFAULT_MODEL, _ = load_policy(ROOT / 'candidate_policy.pt', 'cpu')
SPECIALIST_MODEL, _ = load_policy(ROOT / 'opponent_routed_policy.pt', 'cpu')

def _search(model):
    return InformationSetSearchRuntime(model, DECK, api=CG_API, opponent_hypotheses=HYPOTHESES, basic_pokemon_ids=CATALOG['basic_pokemon_ids'], config=InformationSetRuntimeConfig.from_mapping(RUNTIME_CONFIG), device='cpu', rule_profile=RULE_PROFILE)
_POLICY = OpponentClassifiedPolicyRouterRuntime(default=_search(DEFAULT_MODEL), specialist=_search(SPECIALIST_MODEL), hypotheses=HYPOTHESES, config=OpponentDeckClassifierConfig.from_mapping(CLASSIFIER_CONFIG))

def agent(obs_dict: dict) -> list[int]:
    if not obs_dict.get('select') is None:
        pass
    else:
        _POLICY.reset(DECK)
        return list(DECK)
    action, _ = _POLICY.decide(obs_dict)
    return [int(index) for index in action]
