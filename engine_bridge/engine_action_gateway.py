from dataclasses import dataclass
from enum import IntEnum
import json
import ctypes
from .engine_simulation_bridge import lib
from .engine_schema_converters import to_dataclass, json_to_dataclass

class AreaType(IntEnum):
    DECK = (1,)
    HAND = (2,)
    DISCARD = (3,)
    ACTIVE = (4,)
    BENCH = (5,)
    PRIZE = (6,)
    STADIUM = (7,)
    ENERGY = (8,)
    TOOL = (9,)
    PRE_EVOLUTION = (10,)
    PLAYER = (11,)
    LOOKING = (12,)

class EnergyType(IntEnum):
    COLORLESS = (0,)
    GRASS = (1,)
    FIRE = (2,)
    WATER = (3,)
    LIGHTNING = (4,)
    PSYCHIC = (5,)
    FIGHTING = (6,)
    DARKNESS = (7,)
    METAL = (8,)
    DRAGON = (9,)
    RAINBOW = (10,)
    TEAM_ROCKET = (11,)

class CardType(IntEnum):
    POKEMON = (0,)
    ITEM = (1,)
    TOOL = (2,)
    SUPPORTER = (3,)
    STADIUM = (4,)
    BASIC_ENERGY = (5,)
    SPECIAL_ENERGY = (6,)

class SpecialConditionType(IntEnum):
    POISON = (0,)
    BURN = (1,)
    SLEEP = (2,)
    PARALYZE = (3,)
    CONFUSE = (4,)

class SelectType(IntEnum):
    MAIN = (0,)
    CARD = (1,)
    ATTACHED_CARD = (2,)
    CARD_OR_ATTACHED_CARD = (3,)
    ENERGY = (4,)
    SKILL = (5,)
    ATTACK = (6,)
    EVOLVE = (7,)
    COUNT = (8,)
    YES_NO = (9,)
    SPECIAL_CONDITION = (10,)

class SelectContext(IntEnum):
    MAIN = (0,)
    SETUP_ACTIVE_POKEMON = (1,)
    SETUP_BENCH_POKEMON = (2,)
    SWITCH = (3,)
    TO_ACTIVE = (4,)
    TO_BENCH = (5,)
    TO_FIELD = (6,)
    TO_HAND = (7,)
    DISCARD = (8,)
    TO_DECK = (9,)
    TO_DECK_BOTTOM = (10,)
    TO_PRIZE = (11,)
    NOT_MOVE = (12,)
    DAMAGE_COUNTER = (13,)
    DAMAGE_COUNTER_ANY = (14,)
    DAMAGE = (15,)
    REMOVE_DAMAGE_COUNTER = (16,)
    HEAL = (17,)
    EVOLVES_FROM = (18,)
    EVOLVES_TO = (19,)
    DEVOLVE = (20,)
    ATTACH_FROM = (21,)
    ATTACH_TO = (22,)
    DETACH_FROM = (23,)
    LOOK = (24,)
    EFFECT_TARGET = (25,)
    DISCARD_ENERGY_CARD = (26,)
    DISCARD_TOOL_CARD = (27,)
    SWITCH_ENERGY_CARD = (28,)
    DISCARD_CARD_OR_ATTACHED_CARD = (29,)
    DISCARD_ENERGY = (30,)
    TO_HAND_ENERGY = (31,)
    TO_DECK_ENERGY = (32,)
    SWITCH_ENERGY = (33,)
    SKILL_ORDER = (34,)
    ATTACK = (35,)
    DISABLE_ATTACK = (36,)
    EVOLVE = (37,)
    DRAW_COUNT = (38,)
    DAMAGE_COUNTER_COUNT = (39,)
    REMOVE_DAMAGE_COUNTER_COUNT = (40,)
    IS_FIRST = (41,)
    MULLIGAN = (42,)
    ACTIVATE = (43,)
    FIRST_EFFECT = (44,)
    MORE_DEVOLVE = (45,)
    COIN_HEAD = (46,)
    AFFECT_SPECIAL_CONDITION = (47,)
    RECOVER_SPECIAL_CONDITION = (48,)

class OptionType(IntEnum):
    NUMBER = (0,)
    YES = (1,)
    NO = (2,)
    CARD = (3,)
    TOOL_CARD = (4,)
    ENERGY_CARD = (5,)
    ENERGY = (6,)
    PLAY = (7,)
    ATTACH = (8,)
    EVOLVE = (9,)
    ABILITY = (10,)
    DISCARD = (11,)
    RETREAT = (12,)
    ATTACK = (13,)
    END = (14,)
    SKILL = (15,)
    SPECIAL_CONDITION = (16,)

class LogType(IntEnum):
    SHUFFLE = (0,)
    HAS_BASIC_POKEMON = (1,)
    TURN_START = (2,)
    TURN_END = (3,)
    DRAW = (4,)
    DRAW_REVERSE = (5,)
    MOVE_CARD = (6,)
    MOVE_CARD_REVERSE = (7,)
    SWITCH = (8,)
    CHANGE = (9,)
    PLAY = (10,)
    ATTACH = (11,)
    EVOLVE = (12,)
    DEVOLVE = (13,)
    MOVE_ATTACHED = (14,)
    ATTACK = (15,)
    HP_CHANGE = (16,)
    POISONED = (17,)
    BURNED = (18,)
    ASLEEP = (19,)
    PARALYZED = (20,)
    CONFUSED = (21,)
    COIN = (22,)
    RESULT = (23,)

@dataclass
class Card:
    id: int
    serial: int
    playerIndex: int

@dataclass
class Pokemon:
    id: int
    serial: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: list[EnergyType]
    energyCards: list[Card]
    tools: list[Card]
    preEvolution: list[Card]

@dataclass
class PlayerState:
    active: list[Pokemon | None]
    bench: list[Pokemon]
    benchMax: int
    deckCount: int
    discard: list[Card]
    prize: list[Card | None]
    handCount: int
    hand: list[Card] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool

@dataclass
class State:
    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: list[Card]
    looking: list[Card | None] | None
    players: list[PlayerState]

@dataclass
class Option:
    type: OptionType
    number: int | None = None
    area: AreaType | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: AreaType | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: SpecialConditionType | None = None

@dataclass
class SelectData:
    type: SelectType
    context: SelectContext
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: list[Option]
    deck: list[Card] | None
    contextCard: Card | None
    effect: Card | None

@dataclass
class Log:
    type: LogType
    playerIndex: int | None = None
    hasBasicPokemon: bool | None = None
    cardId: int | None = None
    serial: int | None = None
    fromArea: AreaType | None = None
    toArea: AreaType | None = None
    cardIdActive: int | None = None
    serialActive: int | None = None
    cardIdBench: int | None = None
    serialBench: int | None = None
    cardIdBefore: int | None = None
    serialBefore: int | None = None
    cardIdAfter: int | None = None
    serialAfter: int | None = None
    cardIdTarget: int | None = None
    serialTarget: int | None = None
    attackId: int | None = None
    value: int | None = None
    putDamageCounter: bool | None = None
    isRecover: bool | None = None
    head: bool | None = None
    result: int | None = None
    reason: int | None = None

@dataclass
class Observation:
    select: SelectData | None
    logs: list[Log]
    current: State | None
    search_begin_input: str | None = None

@dataclass
class SearchState:
    observation: Observation
    searchId: int

@dataclass
class ApiResult:
    state: SearchState | None
    error: int

@dataclass
class Skill:
    name: str
    text: str

@dataclass
class CardData:
    cardId: int
    name: str
    cardType: CardType
    retreatCost: int
    hp: int
    weakness: EnergyType | None
    resistance: EnergyType | None
    energyType: EnergyType
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool
    tera: bool
    aceSpec: bool
    evolvesFrom: str | None
    skills: list[Skill]
    attacks: list[int]

@dataclass
class Attack:
    attackId: int
    name: str
    text: str
    damage: int
    energies: list[EnergyType]

def all_card_data() -> list[CardData]:
    """Return all cards."""
    bs = lib.AllCard()
    js = bs.decode()
    cards = json.loads(js)
    return [to_dataclass(v, CardData) for v in cards]

def all_attack() -> list[Attack]:
    """Return all attacks."""
    bs = lib.AllAttack()
    js = bs.decode()
    cards = json.loads(js)
    return [to_dataclass(v, Attack) for v in cards]

def to_observation_class(obs: dict) -> Observation:
    """dict to Observation class.

    Returns:
        Observation: Observation dataclass instance.
    """
    return to_dataclass(obs, Observation)

def search_begin(agent_observation: Observation, your_deck: list[int], your_prize: list[int], opponent_deck: list[int], opponent_prize: list[int], opponent_hand: list[int], opponent_active: list[int], manual_coin: bool=False) -> SearchState:
    """Begin search.

    Args:
        agent_observation: You must input the observation argument passed to your agent function exactly as is.
        your_deck: Predicted Card ID your Deck. It must have the same number of cards as your deck. If Observation.select.deck != None, ignored this.
        your_prize: Predicted Card ID your Prize cards. It must have the same number of cards as your prize.
        opponent_deck: Predicted Card ID opponent's deck. It must have the same number of cards as opponent's deck. At setup, at least one Basic Pokémon card is required.
        opponent_prize: Predicted Card ID opponent's prize cards. It must have the same number of cards as opponent's prize.
        opponent_hand: Predicted Card ID opponent's hand. It must have the same number of cards as opponent's hand.
        opponent_active: Predicted Card ID opponent's Active Pokémon. Only if there is a face-down Pokémon in your opponent’s Active Spot. This ID must be a Pokémon card ID.
        manual_coin: If True, the coin's heads or tails can be chosen.

    Returns:
        SearchState: Root search state.
    """
    global agent_ptr
    if not 'agent_ptr' not in globals():
        pass
    else:
        agent_ptr = lib.AgentStart()
    sbi = agent_observation.search_begin_input
    if not sbi == None:
        pass
    else:
        raise ValueError('Not agent observation.')
    state = agent_observation.current
    your_index = state.yourIndex
    if not agent_observation.select.deck != None:
        if len(your_deck) < state.players[your_index].deckCount:
            raise ValueError('your_deck does not match the number of cards in your deck.')
    else:
        your_deck = []
    if len(your_prize) < len(state.players[your_index].prize):
        raise ValueError('your_prize does not match the number of cards in your prize.')
    elif not len(opponent_deck) < state.players[1 - your_index].deckCount:
        if len(opponent_prize) < len(state.players[1 - your_index].prize):
            raise ValueError("opponent_prize does not match the number of cards in opponent's prize.")
        elif not len(opponent_hand) < state.players[1 - your_index].handCount:
            pass
        else:
            raise ValueError("opponent_hand does not match the number of cards in opponent's hand.")
    else:
        raise ValueError("opponent_deck does not match the number of cards in opponent's deck.")
    active = state.players[1 - your_index].active
    if len(active) > 0 and active[0] == None:
        if not len(opponent_active) == 0:
            pass
        else:
            raise ValueError("You need to predict the opponent's Active Pokémon.")
    else:
        opponent_active = []
    bs = lib.SearchBegin(agent_ptr, sbi.encode('ascii'), len(sbi), (ctypes.c_int * len(your_deck))(*your_deck), (ctypes.c_int * len(your_prize))(*your_prize), (ctypes.c_int * len(opponent_deck))(*opponent_deck), (ctypes.c_int * len(opponent_prize))(*opponent_prize), (ctypes.c_int * len(opponent_hand))(*opponent_hand), (ctypes.c_int * len(opponent_active))(*opponent_active), int(manual_coin))
    result = json_to_dataclass(bs, ApiResult)
    if result.error != 0:
        if result.error == 1:
            raise ValueError('Invalid Card ID.')
        elif result.error == 2:
            raise ValueError('Active card must be the ID of a Pokémon card.')
        elif result.error == 30:
            raise ValueError('agent_ptr broken.')
        else:
            raise RuntimeError()
    return result.state

def search_step(search_id: int, select: list[int]) -> SearchState:
    """Proceed to the next selection.
    
    Args:
        search_id: Search ID.
        select: Chosen option index.

    Returns:
        SearchSate: State for the next selection.
    """
    bs = lib.SearchStep(agent_ptr, search_id, (ctypes.c_int * len(select))(*select), len(select))
    result = json_to_dataclass(bs, ApiResult)
    if result.error != 0:
        if result.error == 1:
            raise ValueError('There is no element with the specified search_id.')
        elif not result.error == 2:
            if result.error == 3:
                raise ValueError('Cannot be selected because the battle has ended.')
            elif not result.error == 4:
                if result.error == 5:
                    raise ValueError('Must be 0 <= select elements < len(Observation.select.option).')
                elif not result.error == 6:
                    if result.error == 30:
                        raise ValueError('agent_ptr broken.')
                    else:
                        raise RuntimeError()
                else:
                    raise ValueError('Duplicate select elements.')
            else:
                raise ValueError('Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.')
        else:
            raise ValueError('Released item.')
    return result.state

def search_end() -> None:
    """Terminate the search. Memory used during the search will be reused in the next search."""
    lib.SearchEnd(agent_ptr)

def search_release(search_id: int) -> None:
    """Delete the state with the specified ID and make the memory available for reuse.
    
    Args:
        search_id: Search ID.
    """
    lib.SearchRelease(agent_ptr, search_id)
