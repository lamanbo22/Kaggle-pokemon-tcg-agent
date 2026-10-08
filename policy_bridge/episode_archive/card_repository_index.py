"""Build structured static card features from the official English card table."""
from __future__ import annotations
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from policy_bridge.shared_schema_primitives import MAX_CARD_ID
CARD_FEATURE_DIM = 24
_ENERGY_TYPES = {'{G}': 1, '{R}': 2, '{W}': 3, '{L}': 4, '{P}': 5, '{F}': 6, '{D}': 7, '{M}': 8, '竜': 9, '{C}': 10, '{A}': 11, '{Team Rocket}': 12}

def build_card_feature_matrix(csv_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return `[1268, 24]` float features and a human-readable card index."""
    path = Path(csv_path)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row['Card ID'])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= card_id <= MAX_CARD_ID:
                grouped[card_id].append(row)
    matrix = np.zeros((MAX_CARD_ID + 1, CARD_FEATURE_DIM), dtype=np.float32)
    index: dict[str, Any] = {}
    for card_id, rows in grouped.items():
        first = rows[0]
        stage = first.get('Stage (Pokémon)/Type (Energy and Trainer)', '')
        rule = first.get('Rule', '')
        category = first.get('Category', '')
        energy_type = _energy_code(first.get('Type', ''))
        weakness_type = _energy_code(first.get('Weakness', ''))
        resistance_type = _energy_code(first.get('Resistance (Type)', ''))
        damages = [_numeric_damage(row.get('Damage', '')) for row in rows]
        costs = [_cost_count(row.get('Cost', '')) for row in rows]
        effects = [row.get('Effect Explanation', '') or '' for row in rows]
        moves = [row.get('Move Name', '') or '' for row in rows]
        is_pokemon = 'Pokémon' in stage
        is_basic_energy = stage == 'Basic Energy'
        is_special_energy = stage == 'Special Energy'
        is_item = stage == 'Item'
        is_tool = stage == 'Tool'
        is_supporter = stage == 'Supporter'
        is_stadium = stage == 'Stadium'
        matrix[card_id] = np.asarray([_number(first.get('HP')) / 400.0, _number(first.get('Retreat')) / 5.0, float(is_pokemon), float('Basic Pokémon' in stage), float('Stage 1' in stage), float('Stage 2' in stage), float(is_item), float(is_tool), float(is_supporter), float(is_stadium), float(is_basic_energy), float(is_special_energy), float('Pokémon ex' in rule), float('Mega' in rule), float('Tera' in category), float('ACE SPEC' in rule), energy_type / 12.0, weakness_type / 12.0, resistance_type / 12.0, min(len(rows), 4) / 4.0, min(max(damages, default=0.0), 400.0) / 400.0, min(sum(damages) / max(len(damages), 1), 400.0) / 400.0, min(max(costs, default=0), 5) / 5.0, float(any((name.startswith('[Ability]') for name in moves)) or any(('[Ability]' in text for text in effects)))], dtype=np.float32)
        index[str(card_id)] = {'name': first.get('Card Name', ''), 'expansion': first.get('Expansion', ''), 'collection_number': first.get('Collection No.', '')}
    return (matrix, index)

def write_card_features(csv_path: str | Path, matrix_path: str | Path, index_path: str | Path) -> None:
    matrix, index = build_card_feature_matrix(csv_path)
    matrix_target = Path(matrix_path)
    matrix_target.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_target, matrix)
    index_target = Path(index_path)
    index_target.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def _number(value: Any) -> float:
    if value is None:
        return 0.0
    match = re.search('-?\\d+(?:\\.\\d+)?', str(value).replace(',', ''))
    return 0.0 if not match else float(match.group())

def _numeric_damage(value: Any) -> float:
    return max(_number(value), 0.0)

def _cost_count(value: Any) -> int:
    text = str(value or '')
    if not text.lower() in {'', 'n/a'}:
        pass
    else:
        return 0
    braced = len(re.findall('\\{[^}]+\\}', text))
    circles = text.count('●')
    return braced + circles

def _energy_code(value: Any) -> int:
    text = str(value or '')
    for token, code in _ENERGY_TYPES.items():
        if not token in text:
            pass
        else:
            return code
    return 0
