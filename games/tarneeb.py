import sys
import numpy as np
from dataclasses import dataclass
from enum import Enum

sys.path.append("..")
from game import Game

# ─── Constants ───────────────────────────────────────────────────────────────

GAME_OVER_SCORE = 31
# Scores can go negative when a team fails to make their bid.
# We encode them in a [−SCORE_OFFSET, SCORE_OFFSET + GAME_OVER_SCORE] window
# shifted to [0, 1] so all state values stay in range.
SCORE_OFFSET = 50          # max expected deficit before winning
SCORE_RANGE  = SCORE_OFFSET + GAME_OVER_SCORE + SCORE_OFFSET  # 131
NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = 52
NUM_PLAYERS = 4
_MAX_RANDOM_SEED = 2 ** 31  # upper bound for np.random.RandomState seed

# Action space layout  (89 total)
# Indices  0-51 : play card   → suit_idx * 13 + (number - 1)
# Indices 52-86 : bid action  → 52 + (value - 7) * 5 + suit_idx
#                               value 7-13, suit_idx 0-3 = H/D/C/S, 4 = suns (no-trump)
# Index   87    : PASS
# Index   88    : DOUBLE
NUM_CARD_ACTIONS = 52
NUM_BID_ACTIONS = 35   # 7 bid values × 5 bid suits
ACTION_PASS = 87
ACTION_DOUBLE = 88
NUM_ACTIONS = 89

# State shape: (NUM_SUITS, NUM_RANKS, NUM_CHANNELS)  i.e. (4, 13, 93)
# All values are in [0, 1].
#
# Channel layout
# ──────────────
#   0 –  3 : which player (0-3) is holding this card
#   4 –  7 : current trick position 0-3 played card
#   8 – 11 : current player one-hot              (broadcast over 4×13)
#  12 – 15 : trump suit one-hot H/D/C/S          (broadcast)
#       16 : no-trump (suns) game flag            (broadcast)
#       17 : is-bidding-phase flag                (broadcast)
#  18 – 24 : current high bid one-hot 7–13        (broadcast)
#  25 – 28 : bidder index one-hot 0-3             (broadcast)
#       29 : no-bidder flag                       (broadcast)
#  30 – 34 : passes count one-hot 0-4             (broadcast)
#  35 – 38 : double_by index one-hot 0-3          (broadcast)
#       39 : no-double flag                       (broadcast)
#       40 : team-0 cumulative score / 31         (broadcast)
#       41 : team-1 cumulative score / 31         (broadcast)
#       42 : team-0 round-trick score / 13        (broadcast)
#       43 : team-1 round-trick score / 13        (broadcast)
#       44 : round number / 50 (capped at 1)      (broadcast)
#  45 – 92 : per-player bid (4 players × 12 ch):
#            base = 45 + p*12
#            base+0..6  bid value one-hot 7-13
#            base+7..10 bid suit one-hot H/D/C/S
#            base+11    suns bid flag
NUM_CHANNELS = 93


# ─── Suit / Card definitions ─────────────────────────────────────────────────

class Suit(Enum):
    HEARTS   = (3, "Hearts",   "H")
    DIAMONDS = (2, "Diamonds", "D")
    CLUBS    = (1, "Clubs",    "C")
    SPADES   = (4, "Spades",   "S")

    def __init__(self, numeric_value, full_name, abbv):
        self._value_ = numeric_value
        self.full_name = full_name
        self.abbv = abbv


# Canonical ordering used throughout encoding/decoding
SUIT_ORDER = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]


@dataclass(frozen=True)
class Card:
    suit: Suit
    number: int  # 1 (Ace) – 13 (King)

    def value(self) -> int:
        """Ace is high (13), 2-K are 1-12."""
        return (self.number - 1) if self.number > 1 else 13

    def __str__(self) -> str:
        face = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(self.number, str(self.number))
        return f"{self.suit.abbv}{face}"


# ─── Card ↔ index helpers ────────────────────────────────────────────────────

def _card_coords(card: Card):
    """Returns (suit_idx, rank_idx) for the 4×13 state grid."""
    return SUIT_ORDER.index(card.suit), card.number - 1


def _card_to_idx(card: Card) -> int:
    """Card → action index 0-51."""
    return SUIT_ORDER.index(card.suit) * NUM_RANKS + (card.number - 1)


def _card_from_idx(idx: int) -> Card:
    """Action index 0-51 → Card."""
    return Card(SUIT_ORDER[idx // NUM_RANKS], idx % NUM_RANKS + 1)


def _create_deck():
    return [Card(suit, num) for suit in SUIT_ORDER for num in range(1, NUM_RANKS + 1)]


# ─── Tarneeb Game ─────────────────────────────────────────────────────────────

class Tarneeb(Game):
    """
    4-player team trick-taking card game.
    Players 0 & 2 form Team 0; players 1 & 3 form Team 1.

    Phases
    ------
    Bidding : Players bid for the number of tricks (7-13), pass, or double the
              current high bid.  Bidding closes when 3 consecutive passes follow
              the last bid (bidder wins), or all 4 players have each placed a bid,
              or all 4 pass without any bid (no-trump round).
    Playing : Players take turns playing cards.  Must follow the led suit if
              possible.  Trump cards beat non-trump; highest on-suit card wins
              the trick otherwise.  Trick winner leads the next.

    Scoring
    -------
    After all 13 tricks the round ends.  The bidding team scores their collected
    tricks if >= their bid; otherwise they lose the bid value.  Doubling
    multiplies gains/losses by 2.  First team to reach 31 wins.

    State encoding
    --------------
    Shape: (4, 13, 93)  — suits × ranks × feature channels (all values in [0,1]).

    Action encoding
    ---------------
    Shape: (89,) boolean mask.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def get_num_players(self):
        return NUM_PLAYERS

    def get_initial_state(self):
        deck = _create_deck()
        indices = np.random.permutation(NUM_CARDS)
        deck = [deck[i] for i in indices]
        state = {
            'holding_cards': [deck[i * 13:(i + 1) * 13] for i in range(NUM_PLAYERS)],
            'played_cards': [],
            'trump_suit': None,
            'suit_selected': False,
            'passes_count': 0,
            'double_by': None,
            'score': (0, 0),
            'round_score': (0, 0),
            'round_num': 1,
            'current_player': 0,
            'current_high_bid': 6,  # minimum valid bid is 7
            'bidder': None,
            'bids': [None] * NUM_PLAYERS,
        }
        return self._encode(state)

    def get_player(self, s):
        return int(np.argmax(s[0, 0, 8:12]))

    def get_available_actions(self, s):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        current_player = int(np.argmax(s[0, 0, 8:12]))
        is_bidding = s[0, 0, 17] > 0.5

        if is_bidding:
            # ── Decode current high bid ──
            bid_enc = s[0, 0, 18:25]
            if bid_enc.max() > 0.5:
                current_high_bid = 7 + int(np.argmax(bid_enc))
            else:
                current_high_bid = 6  # no bid placed yet

            # ── Is the current top bid "suns" (no-trump)? ──
            bidder_enc = s[0, 0, 25:30]
            bidder_none = bidder_enc[4] > 0.5
            bidder = None if bidder_none else int(np.argmax(bidder_enc[:4]))

            current_bid_is_suns = False
            if bidder is not None:
                base = 45 + bidder * 12
                current_bid_is_suns = s[0, 0, base + 11] > 0.5

            # ── Bid actions ──
            for v_idx in range(7):
                v = 7 + v_idx
                for suit_idx in range(5):
                    is_suns = (suit_idx == 4)
                    suns_ok = (v == current_high_bid and is_suns
                               and not current_bid_is_suns)
                    if v > current_high_bid or suns_ok:
                        mask[52 + v_idx * 5 + suit_idx] = True

            # ── PASS is always valid during bidding ──
            mask[ACTION_PASS] = True

            # ── DOUBLE: only if someone has bid and no double yet ──
            if bidder is not None and s[0, 0, 39] > 0.5:
                mask[ACTION_DOUBLE] = True

        else:
            # ── Playing phase ──
            hand_mask = s[:, :, current_player] > 0.5  # (4, 13) bool

            # Find the led suit from the first card in the current trick
            led_suit = None
            for si in range(NUM_SUITS):
                if s[si, :, 4].any():
                    led_suit = si
                    break

            if led_suit is not None and hand_mask[led_suit].any():
                # Must follow suit
                for ri in range(NUM_RANKS):
                    if hand_mask[led_suit, ri]:
                        mask[led_suit * NUM_RANKS + ri] = True
            else:
                # May play any card
                for si in range(NUM_SUITS):
                    for ri in range(NUM_RANKS):
                        if hand_mask[si, ri]:
                            mask[si * NUM_RANKS + ri] = True

        return mask

    def check_game_over(self, s):
        score0 = s[0, 0, 40] * SCORE_RANGE - SCORE_OFFSET
        score1 = s[0, 0, 41] * SCORE_RANGE - SCORE_OFFSET
        if score0 >= GAME_OVER_SCORE - 0.5 or score1 >= GAME_OVER_SCORE - 0.5:
            if score0 > score1:
                return np.array([1.0, -1.0,  1.0, -1.0], dtype=np.float32)
            elif score1 > score0:
                return np.array([-1.0,  1.0, -1.0,  1.0], dtype=np.float32)
            else:
                return np.zeros(NUM_PLAYERS, dtype=np.float32)
        return None

    def take_action(self, s, a):
        action_idx = int(np.argwhere(a).flatten()[0])
        state = self._decode(s)

        if state['suit_selected']:
            # Playing phase — action is a card
            card = _card_from_idx(action_idx)
            new_state = self._apply_card(state, card)
        elif action_idx == ACTION_PASS:
            new_state = self._apply_pass(state)
        elif action_idx == ACTION_DOUBLE:
            new_state = self._apply_double(state)
        else:
            # Bid action
            bid_offset = action_idx - 52
            v_idx = bid_offset // 5
            suit_idx = bid_offset % 5
            bid_value = 7 + v_idx
            bid_suit = SUIT_ORDER[suit_idx] if suit_idx < 4 else None
            new_state = self._apply_bid(state, bid_value, bid_suit)

        return self._encode(new_state)

    def visualize(self, s):
        state = self._decode(s)
        print("=" * 60)
        print(f"Round {state['round_num']}  |  "
              f"Score  Team-0: {state['score'][0]}  Team-1: {state['score'][1]}")
        for p in range(NUM_PLAYERS):
            hand = sorted(state['holding_cards'][p], key=_card_to_idx)
            team = p % 2
            print(f"  Player {p} (Team {team}): {' '.join(str(c) for c in hand)}")
        if state['suit_selected']:
            trump_str = (state['trump_suit'].abbv if state['trump_suit']
                         else "Suns (no-trump)")
            print(f"Phase: Playing  |  Trump: {trump_str}")
            trick = state['played_cards']
            if trick:
                num_in = len(trick)
                leader = (state['current_player'] - num_in + NUM_PLAYERS) % NUM_PLAYERS
                print(f"  Current trick: {' '.join(str(c) for c in trick)}"
                      f"  (led by P{leader})")
            print(f"  Round tricks  Team-0: {state['round_score'][0]}"
                  f"  Team-1: {state['round_score'][1]}")
        else:
            bidder_str = str(state['bidder']) if state['bidder'] is not None else "-"
            print(f"Phase: Bidding  |  High bid: {state['current_high_bid']}"
                  f"  by P{bidder_str}  Passes: {state['passes_count']}")
            bids_str = [(str(b) if b else "pass") for b in state['bids']]
            print(f"  Bids: {bids_str}")
            if state['double_by'] is not None:
                print(f"  DOUBLED by P{state['double_by']}")
        print(f"Current player: {state['current_player']}")

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode(self, state: dict) -> np.ndarray:
        s = np.zeros((NUM_SUITS, NUM_RANKS, NUM_CHANNELS), dtype=np.float32)

        # Player hands (channels 0-3)
        for p, hand in enumerate(state['holding_cards']):
            for card in hand:
                si, ri = _card_coords(card)
                s[si, ri, p] = 1.0

        # Current trick cards by position (channels 4-7)
        for pos, card in enumerate(state['played_cards']):
            si, ri = _card_coords(card)
            s[si, ri, 4 + pos] = 1.0

        # Current player one-hot (channels 8-11, broadcast)
        s[:, :, 8 + state['current_player']] = 1.0

        # Trump suit / phase flags (channels 12-17, broadcast)
        if state['suit_selected']:
            if state['trump_suit'] is not None:
                s[:, :, 12 + SUIT_ORDER.index(state['trump_suit'])] = 1.0
            else:
                s[:, :, 16] = 1.0  # no-trump / suns
        else:
            s[:, :, 17] = 1.0  # bidding phase

        # Current high bid (channels 18-24, broadcast)
        if state['current_high_bid'] >= 7:
            s[:, :, 18 + (state['current_high_bid'] - 7)] = 1.0

        # Bidder (channels 25-29, broadcast)
        if state['bidder'] is not None:
            s[:, :, 25 + state['bidder']] = 1.0
        else:
            s[:, :, 29] = 1.0  # no-bidder flag

        # Passes count (channels 30-34, broadcast)
        s[:, :, 30 + min(state['passes_count'], 4)] = 1.0

        # Double-by (channels 35-39, broadcast)
        if state['double_by'] is not None:
            s[:, :, 35 + state['double_by']] = 1.0
        else:
            s[:, :, 39] = 1.0  # no-double flag

        # Scores (channels 40-44, broadcast)
        s[:, :, 40] = min(max((state['score'][0] + SCORE_OFFSET) / SCORE_RANGE, 0.0), 1.0)
        s[:, :, 41] = min(max((state['score'][1] + SCORE_OFFSET) / SCORE_RANGE, 0.0), 1.0)
        s[:, :, 42] = min(state['round_score'][0] / 13.0, 1.0)
        s[:, :, 43] = min(state['round_score'][1] / 13.0, 1.0)
        s[:, :, 44] = min(state['round_num'] / 50.0, 1.0)

        # Per-player bid info (channels 45-92, broadcast)
        for p in range(NUM_PLAYERS):
            base = 45 + p * 12
            bid_p = state['bids'][p]
            if bid_p is not None:
                v, suit = bid_p
                if 7 <= v <= 13:
                    s[:, :, base + (v - 7)] = 1.0
                if suit is not None:
                    s[:, :, base + 7 + SUIT_ORDER.index(suit)] = 1.0
                else:
                    s[:, :, base + 11] = 1.0  # suns

        return s

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _decode(self, s: np.ndarray) -> dict:
        # Player hands
        holding_cards = [[] for _ in range(NUM_PLAYERS)]
        for p in range(NUM_PLAYERS):
            for si in range(NUM_SUITS):
                for ri in range(NUM_RANKS):
                    if s[si, ri, p] > 0.5:
                        holding_cards[p].append(Card(SUIT_ORDER[si], ri + 1))

        # Current trick (ordered by trick position 0-3)
        played_cards = []
        for pos in range(4):
            found = None
            for si in range(NUM_SUITS):
                for ri in range(NUM_RANKS):
                    if s[si, ri, 4 + pos] > 0.5:
                        found = Card(SUIT_ORDER[si], ri + 1)
                        break
                if found:
                    break
            if found:
                played_cards.append(found)
            else:
                break  # no more cards at this position

        # Current player
        current_player = int(np.argmax(s[0, 0, 8:12]))

        # Suit / phase
        is_bidding = s[0, 0, 17] > 0.5
        suit_selected = not is_bidding
        if suit_selected:
            trump_enc = s[0, 0, 12:17]
            ti = int(np.argmax(trump_enc))
            trump_suit = SUIT_ORDER[ti] if ti < 4 else None
        else:
            trump_suit = None

        # Current high bid
        bid_enc = s[0, 0, 18:25]
        if bid_enc.max() > 0.5:
            current_high_bid = 7 + int(np.argmax(bid_enc))
        else:
            current_high_bid = 6  # no bid placed yet

        # Bidder
        bidder_enc = s[0, 0, 25:30]
        bi = int(np.argmax(bidder_enc))
        bidder = None if bi == 4 else bi

        # Passes count
        passes_count = int(np.argmax(s[0, 0, 30:35]))

        # Double-by
        double_enc = s[0, 0, 35:40]
        di = int(np.argmax(double_enc))
        double_by = None if di == 4 else di

        # Scores
        score0 = round(s[0, 0, 40] * SCORE_RANGE - SCORE_OFFSET)
        score1 = round(s[0, 0, 41] * SCORE_RANGE - SCORE_OFFSET)
        rs0    = round(s[0, 0, 42] * 13)
        rs1    = round(s[0, 0, 43] * 13)
        round_num = max(1, round(s[0, 0, 44] * 50))

        # Per-player bids
        bids = [None] * NUM_PLAYERS
        for p in range(NUM_PLAYERS):
            base = 45 + p * 12
            val_enc  = s[0, 0, base:base + 7]
            suit_enc = s[0, 0, base + 7:base + 12]
            if val_enc.max() > 0.5:
                bv = 7 + int(np.argmax(val_enc))
                si_b = int(np.argmax(suit_enc))
                bs = SUIT_ORDER[si_b] if si_b < 4 else None
                bids[p] = (bv, bs)

        return {
            'holding_cards': holding_cards,
            'played_cards': played_cards,
            'trump_suit': trump_suit,
            'suit_selected': suit_selected,
            'passes_count': passes_count,
            'double_by': double_by,
            'score': (int(score0), int(score1)),
            'round_score': (int(rs0), int(rs1)),
            'round_num': int(round_num),
            'current_player': current_player,
            'current_high_bid': current_high_bid,
            'bidder': bidder,
            'bids': bids,
        }

    # ── Bidding actions ───────────────────────────────────────────────────────

    def _apply_pass(self, state: dict) -> dict:
        new_passes = state['passes_count'] + 1
        new_state = dict(state)
        new_state['passes_count'] = new_passes

        if state['bidder'] is None and new_passes == 4:
            # All four passed with no bid → no-trump round
            new_state['suit_selected'] = True
            new_state['trump_suit'] = None
            new_state['current_player'] = 0
        elif state['bidder'] is not None and new_passes == 3:
            # Three consecutive passes after the last bid → bidder wins
            winning_bid = state['bids'][state['bidder']]
            new_state['suit_selected'] = True
            new_state['trump_suit'] = winning_bid[1] if winning_bid else None
            new_state['current_player'] = state['bidder']
        else:
            new_state['current_player'] = (
                (state['current_player'] + 1) % NUM_PLAYERS)

        return new_state

    def _apply_double(self, state: dict) -> dict:
        new_state = dict(state)
        new_state['double_by'] = state['current_player']
        # After a double the bidder acts next (can raise, pass, etc.)
        new_state['current_player'] = state['bidder']
        return new_state

    def _apply_bid(self, state: dict, bid_value: int, bid_suit) -> dict:
        new_bids = list(state['bids'])
        new_bids[state['current_player']] = (bid_value, bid_suit)

        new_state = dict(state)
        new_state['current_high_bid'] = bid_value
        new_state['bidder'] = state['current_player']
        new_state['bids'] = new_bids
        new_state['passes_count'] = 0  # reset consecutive passes

        if all(b is not None for b in new_bids):
            # All four players have submitted a bid → bidding closes
            new_state['suit_selected'] = True
            new_state['trump_suit'] = bid_suit
            # The winning bidder leads the first trick
            new_state['current_player'] = state['current_player']
        else:
            new_state['current_player'] = (
                (state['current_player'] + 1) % NUM_PLAYERS)

        return new_state

    # ── Playing actions ───────────────────────────────────────────────────────

    def _apply_card(self, state: dict, card: Card) -> dict:
        new_holding = [list(h) for h in state['holding_cards']]
        new_holding[state['current_player']].remove(card)
        new_played = list(state['played_cards']) + [card]

        if len(new_played) == 4:
            return self._complete_trick(state, new_holding, new_played)

        # Trick still in progress
        new_state = dict(state)
        new_state['holding_cards'] = new_holding
        new_state['played_cards'] = new_played
        new_state['current_player'] = (
            (state['current_player'] + 1) % NUM_PLAYERS)
        return new_state

    def _complete_trick(self, state: dict, new_holding, played_cards) -> dict:
        """Called after the 4th card of a trick has been played."""
        last_player = state['current_player']
        winner, trick_score = self._calc_trick_winner(
            last_player, played_cards, state['trump_suit'])

        new_round_score = (
            state['round_score'][0] + trick_score[0],
            state['round_score'][1] + trick_score[1],
        )

        base_state = dict(state)
        base_state['holding_cards'] = new_holding
        base_state['played_cards'] = []
        base_state['round_score'] = new_round_score
        base_state['current_player'] = winner

        # Check if the round is over (all hands empty)
        if all(len(h) == 0 for h in new_holding):
            return self._complete_round(base_state)

        return base_state

    def _complete_round(self, state: dict) -> dict:
        """Called after the last trick of a round."""
        new_score = self._calc_round_end_score(state, state['round_score'])

        # Encode new cumulative score so check_game_over can detect game over
        state_with_score = dict(state)
        state_with_score['score'] = new_score

        # If the game is now over, keep the terminal state as-is
        if any(ts >= GAME_OVER_SCORE for ts in new_score):
            return state_with_score

        # Otherwise, deal a new round deterministically from the score state
        return self._start_new_round(new_score, state['round_num'] + 1)

    def _start_new_round(self, new_score: tuple, round_num: int) -> dict:
        """Deal a new hand.  The shuffle is seeded deterministically so that
        take_action is a pure function (same state + action → same successor)."""
        seed = (int(new_score[0]) * 10000
                + int(new_score[1]) * 100
                + int(round_num)) % _MAX_RANDOM_SEED
        rng = np.random.RandomState(seed)
        deck = _create_deck()
        deck = [deck[i] for i in rng.permutation(NUM_CARDS)]
        return {
            'holding_cards': [deck[i * 13:(i + 1) * 13] for i in range(NUM_PLAYERS)],
            'played_cards': [],
            'trump_suit': None,
            'suit_selected': False,
            'passes_count': 0,
            'double_by': None,
            'score': new_score,
            'round_score': (0, 0),
            'round_num': round_num,
            'current_player': 0,
            'current_high_bid': 6,
            'bidder': None,
            'bids': [None] * NUM_PLAYERS,
        }

    # ── Scoring helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _calc_trick_winner(last_player: int, played_cards: list, trump_suit):
        """Returns (winning_player_idx, (team0_tricks, team1_tricks))."""
        first_player = (last_player - 3 + NUM_PLAYERS) % NUM_PLAYERS
        winning_card = played_cards[0]
        winning_player = first_player

        for i, card in enumerate(played_cards[1:], start=1):
            player_i = (first_player + i) % NUM_PLAYERS
            beats = (
                card.suit == winning_card.suit
                and card.value() > winning_card.value()
            ) or (
                trump_suit is not None
                and card.suit == trump_suit
                and winning_card.suit != trump_suit
            )
            if beats:
                winning_card = card
                winning_player = player_i

        if winning_player % 2 == 0:
            return winning_player, (1, 0)
        return winning_player, (0, 1)

    @staticmethod
    def _calc_round_end_score(state: dict, round_score: tuple) -> tuple:
        """Applies bid/double scoring to produce new cumulative (score0, score1)."""
        caller_idx = state['bidder']
        rt0, rt1 = round_score

        if caller_idx is None:
            # All-pass round: plain trick scoring, no bid penalty
            return (state['score'][0] + rt0, state['score'][1] + rt1)

        caller_team = caller_idx % 2
        call_value  = state['current_high_bid']
        doubled     = state['double_by'] is not None

        if caller_team == 0:
            caller_collected, other_collected = rt0, rt1
        else:
            caller_collected, other_collected = rt1, rt0

        if caller_collected >= call_value:
            caller_delta     = caller_collected * (2 if doubled else 1)
            non_caller_delta = other_collected           # non-caller is NOT doubled on success
        else:
            caller_delta     = -(call_value * (2 if doubled else 1))
            non_caller_delta = other_collected * (2 if doubled else 1)  # non-caller IS doubled on failure

        if caller_team == 0:
            return (state['score'][0] + caller_delta,
                    state['score'][1] + non_caller_delta)
        return (state['score'][0] + non_caller_delta,
                state['score'][1] + caller_delta)
