"""Observation / action tensor layout shared by the C# encoder (sim + mod) and the PyTorch model.

Keep in sync with sim/Encoding/ObsSpec.cs. Bump VERSION on any change.

Each decision is a set of entity tokens plus a set of legal-action tokens:

  token_cat  int32[T, 5]   (token_type, vocab_id, vocab_id2, aux1, aux2)
                           aux2 is always the "slot" (creature slot / owner slot / position)
  token_num  float32[T, F_TOK]
  action_cat int32[A, 3]   (action_type, vocab_id, aux)
  action_ref int32[A, 2]   indices into the token list (-1 = none), e.g. (card, target)
  action_num float32[A, F_ACT]

Token 0 is always the GLOBAL token. The policy scores every action token; the value
heads read the GLOBAL token after the encoder.
"""

VERSION = 1

MAX_TOKENS = 192
MAX_ACTIONS = 128
F_TOK = 32
F_ACT = 8
N_TOK_CAT = 5
N_ACT_CAT = 3
HASH_BUCKETS = 1024  # vocab tail reserved for hashed strings (move ids, option keys)
AUX_VOCAB = 128  # size of the small-enum embedding tables (aux1/aux2/action aux)


class TokenType:
    PAD = 0
    GLOBAL = 1
    HAND_CARD = 2
    DRAW_CARD = 3
    DISCARD_CARD = 4
    EXHAUST_CARD = 5
    DECK_CARD = 6        # master deck (outside combat)
    OFFER_CARD = 7       # card offered by reward / shop / event / selection prompt
    PLAYER = 8
    ENEMY = 9
    POWER = 10
    RELIC = 11
    POTION = 12
    ORB = 13
    MAP_NODE = 14
    OPTION = 15          # event / rest-site / generic option
    OFFER_RELIC = 16
    OFFER_POTION = 17
    HISTORY = 18         # run-history summary tokens ("global memory")
    BOSS = 19            # upcoming boss encounter(s)
    PLAY_PILE = 20
    COUNT = 21


class ActionType:
    PAD = 0
    END_TURN = 1
    PLAY_CARD = 2
    USE_POTION = 3
    DISCARD_POTION = 4
    CHOOSE_OPTION = 5      # event / rest site / ancient option
    CHOOSE_MAP_NODE = 6
    SELECT_CARD = 7        # card selection prompt / card reward pick
    SKIP = 8               # skip reward / confirm selection with current picks / leave
    CLAIM_REWARD = 9       # gold / potion / relic / open card reward
    BUY = 10
    REMOVE_CARD_SERVICE = 11
    PROCEED = 12
    SELECT_RELIC = 13
    CONFIRM = 14
    COUNT = 15


class DecisionType:
    COMBAT = 1
    CARD_SELECT = 2
    CARD_REWARD = 3
    REWARDS = 4
    MAP = 5
    EVENT = 6
    REST = 7
    SHOP = 8
    TREASURE = 9
    RELIC_SELECT = 10
    BUNDLE_SELECT = 11
    OTHER = 12
    COUNT = 13

    STRATEGIC = {CARD_REWARD, REWARDS, MAP, EVENT, REST, SHOP, TREASURE, RELIC_SELECT, BUNDLE_SELECT}
