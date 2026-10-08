"""Feature constants shared by preprocessing, training, and submission inference."""
from __future__ import annotations
from enum import IntEnum
PAD_CARD_ID = 0
MAX_CARD_ID = 1267
MAX_ATTACK_ID = 1556
FEATURE_SEMANTICS_VERSION = 3
OPTION_TYPE_PLAY = 7

class Zone(IntEnum):
    PAD = 0
    HAND = 1
    ACTIVE = 2
    BENCH = 3
    DISCARD = 4
    PRIZE = 5
    STADIUM = 6
    LOOKING = 7
    SELECT_DECK = 8
    ATTACHED_ENERGY = 9
    TOOL = 10
    PRE_EVOLUTION = 11
    EFFECT = 12
    CONTEXT = 13

class EntityKind(IntEnum):
    PAD = 0
    CARD = 1
    POKEMON = 2
    ATTACHMENT = 3
    CONTEXT = 4

class RelativeOwner(IntEnum):
    UNKNOWN = 0
    SELF = 1
    OPPONENT = 2
    NEUTRAL = 3
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12
GLOBAL_CAT_DIM = 5
GLOBAL_NUM_DIM = 16
ENTITY_CAT_DIM = 5
ENTITY_NUM_DIM = 9
EVENT_CAT_DIM = 8
EVENT_NUM_DIM = 3
OPTION_CAT_DIM = 14
OPTION_NUM_DIM = 4
