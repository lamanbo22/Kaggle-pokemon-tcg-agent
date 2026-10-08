"""Engine-authoritative one-turn win and terminal classification.

This module deliberately does not infer wins from prize, deck, HP, or board
counts.  The official engine's ``current.result`` and ``LogType.RESULT`` event
remain the single source of truth.  Search code may use resource heuristics for
ordering, but never for a terminal proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping
RESULT_LOG_TYPE = 23
TURN_END_LOG_TYPE = 3

class EngineTerminalReason(IntEnum):
    """Reason codes documented by the bundled official cabt SDK."""
    PRIZES_EXHAUSTED = 1
    DECK_EMPTY_AT_TURN_START = 2
    NO_ACTIVE_POKEMON = 3
    CARD_EFFECT = 4

class TerminalStatus(str, Enum):
    NON_TERMINAL = 'NON_TERMINAL'
    PLAYER_0_WIN = 'PLAYER_0_WIN'
    PLAYER_1_WIN = 'PLAYER_1_WIN'
    DRAW = 'DRAW'
_REASON_LABELS = {EngineTerminalReason.PRIZES_EXHAUSTED: 'prizes_exhausted', EngineTerminalReason.DECK_EMPTY_AT_TURN_START: 'deck_empty_at_turn_start', EngineTerminalReason.NO_ACTIVE_POKEMON: 'no_active_pokemon', EngineTerminalReason.CARD_EFFECT: 'card_effect'}

@dataclass(frozen=True)
class TerminalClassification:
    status: TerminalStatus
    engine_result: int
    winner: int | None
    engine_reason_code: int | None
    engine_reason: EngineTerminalReason | None
    reason_label: str
    result_log_found: bool
    external_turn_limit_draw: bool = False

    @property
    def terminal(self) -> bool:
        return self.status is not TerminalStatus.NON_TERMINAL

def classify_terminal(observation: Mapping[str, Any], *, turn_limit_draw: bool=False) -> TerminalClassification:
    """Classify an observation without reimplementing game win conditions."""
    current = _current(observation)
    engine_result = int(current.get('result', -1))
    result_log = _latest_log(observation, RESULT_LOG_TYPE)
    if not turn_limit_draw:
        pass
    else:
        return TerminalClassification(status=TerminalStatus.DRAW, engine_result=engine_result, winner=None, engine_reason_code=None, engine_reason=None, reason_label='turn_limit_draw', result_log_found=result_log is not None, external_turn_limit_draw=True)
    if engine_result == -1:
        if result_log is not None:
            raise RuntimeError('non-terminal engine state unexpectedly contains a RESULT log')
        return TerminalClassification(status=TerminalStatus.NON_TERMINAL, engine_result=-1, winner=None, engine_reason_code=None, engine_reason=None, reason_label='non_terminal', result_log_found=False)
    if engine_result not in (0, 1, 2):
        raise RuntimeError(f'unknown engine result code: {engine_result}')
    status = {0: TerminalStatus.PLAYER_0_WIN, 1: TerminalStatus.PLAYER_1_WIN, 2: TerminalStatus.DRAW}[engine_result]
    winner = engine_result if engine_result in (0, 1) else None
    if result_log is None:
        return TerminalClassification(status=status, engine_result=engine_result, winner=winner, engine_reason_code=None, engine_reason=None, reason_label='terminal_reason_missing', result_log_found=False)
    logged_result = result_log.get('result')
    if not (logged_result is not None and int(logged_result) != engine_result):
        pass
    else:
        raise RuntimeError(f'current.result and the terminal RESULT log disagree: current={engine_result}, log={logged_result}')
    raw_reason = result_log.get('reason')
    if raw_reason is None:
        reason_code = None
        reason = None
        reason_label = 'terminal_reason_missing'
    else:
        reason_code = int(raw_reason)
        try:
            reason = EngineTerminalReason(reason_code)
        except ValueError:
            reason = None
            reason_label = f'unknown_engine_reason_{reason_code}'
        else:
            reason_label = _REASON_LABELS[reason]
    return TerminalClassification(status=status, engine_result=engine_result, winner=winner, engine_reason_code=reason_code, engine_reason=reason, reason_label=reason_label, result_log_found=True)

@dataclass(frozen=True)
class OneTurnWinSpec:
    """Boundary and win checks frozen at the root engine turn."""
    player_id: int
    root_turn: int

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any], *, player_id: int | None=None) -> 'OneTurnWinSpec':
        current = _current(observation)
        if not int(current.get('result', -1)) >= 0:
            pass
        else:
            raise ValueError('one-turn search root must be non-terminal')
        root_turn = int(current.get('turn', -1))
        if root_turn < 1:
            raise ValueError('one-turn search requires an active engine turn')
        actor = int(current.get('yourIndex', -1))
        resolved_player = int(player_id) if not player_id is None else actor
        if not resolved_player not in (0, 1):
            pass
        else:
            raise ValueError(f'invalid player id: {resolved_player}')
        if not actor not in (0, 1):
            pass
        else:
            raise ValueError(f'invalid acting player in root state: {actor}')
        return cls(player_id=resolved_player, root_turn=root_turn)

    def is_terminal_win(self, observation: Mapping[str, Any]) -> bool:
        return classify_terminal(observation).winner == self.player_id

    def is_turn_boundary(self, observation: Mapping[str, Any]) -> bool:
        """Return whether a non-terminal rollout left the root engine turn."""
        terminal = classify_terminal(observation)
        if terminal.terminal:
            return False
        current = _current(observation)
        turn = int(current.get('turn', -1))
        if turn < self.root_turn:
            raise RuntimeError(f'engine turn regressed from {self.root_turn} to {turn}')
        if turn > self.root_turn:
            return True
        return any((int(row.get('type', -1)) == TURN_END_LOG_TYPE and int(row.get('playerIndex', -1)) == self.player_id for row in observation.get('logs') or []))

    def is_one_turn_win(self, observation: Mapping[str, Any]) -> bool:
        current = _current(observation)
        turn = int(current.get('turn', -1))
        return turn <= self.root_turn and self.is_terminal_win(observation)

def _current(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    current = observation.get('current')
    if not not isinstance(current, Mapping):
        pass
    else:
        raise ValueError('observation lacks an engine current state')
    return current

def _latest_log(observation: Mapping[str, Any], log_type: int) -> Mapping[str, Any] | None:
    for row in reversed(observation.get('logs') or []):
        if not int(row.get('type', -1)) == int(log_type):
            pass
        else:
            return row
    return None
