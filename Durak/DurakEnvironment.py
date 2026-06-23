from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


# Card representations
CARDS = ['6', '7', '8', '9', '10', 'B', 'Q', 'K', 'A']  # 6-10 + Jack(B), Queen(Q), King(K), Ace(A)
SUITS = ['♠', '♥', '♦', '♣']  # Spades, Hearts, Diamonds, Clubs
TRUMP_VALUES = {card: i for i, card in enumerate(CARDS)}

class DurakEnv(gym.Env):
    """
    A custom environment for a card game.
    
    State Space: 
        - Player hand (encoded)
        - Trump card is visible
        - remember all cards that are successfully defended
        - Deck remaining count
        - Current turn
        
    Action Space:
        - Play a specific card index from hand
        - Draw a card
        - Pass (if applicable)
    """

    def __init__(self, num_players=2, deck_size=36, max_hand_size=6):
        super().__init__()
        
        # --- Configuration ---
        self.num_players = num_players
        self.deck_size = deck_size
        self.max_hand_size = max_hand_size
        self.current_player = 0
        
        # Card index mapping (suit * 9 + rank)
        self.card_map = {}
        idx = 0
        for suit in SUITS:
            for card in CARDS:
                self.card_map[f"{card}{suit}"] = idx
                idx += 1
        
        self.total_cards = len(CARDS) * len(SUITS)  # 36 cards
        
        # Define observation space
        # We'll track: player hand (one-hot), known cards, deck status, current state
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: play card or pass
        # Max possible actions = number of cards + 1 (for pass/defend) 
        # TODO There are multiple cards to be attacked with. Fix this please.
        action_dim = self.total_cards + 1
        self.action_space = spaces.Discrete(action_dim)
        
        # Game state
        self.deck = []
        self.player_hands = [[] for _ in range(n_players)]
        self.table_cards = []  # Cards currently on table (attack/defense pairs)
        self.trump_suit = None
        self.current_attacker = 0
        self.current_defender = 0
        self.game_phase = 'preparation'  # preparation, attack, defend, draw
        self.first_trump_player = None
        self.cards_played_this_round = set()

############################

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.deck = list(range(self.deck_size))
        random.shuffle(self.deck)
        self.hands = [[] for _ in range(self.num_players)]
        self.scores = [0] * self.num_players
        self.current_player = 0
        self.game_over = False
        
        # Deal initial hands
        for _ in range(self.max_hand_size):
            for p in range(self.num_players):
                if self.deck:
                    self.hands[p].append(self.deck.pop())
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.game_over:
            raise RuntimeError("Environment is already done. Call reset().")

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- Game Logic Placeholder ---
        # 1. Validate Action
        if action < len(self.hands[self.current_player]):
            # Player plays a card
            card_idx = action
            card = self.hands[self.current_player].pop(card_idx)
            self._process_played_card(card, self.current_player)
            reward += 1 # Small reward for playing
            
            # Draw replacement if hand size < max
            if len(self.hands[self.current_player]) < self.max_hand_size and self.deck:
                self.hands[self.current_player].append(self.deck.pop())
                
        elif action == len(self.hands[self.current_player]):
            # Draw action (if allowed)
            if self.deck:
                self.hands[self.current_player].append(self.deck.pop())
            else:
                # No cards left to draw, maybe pass forced?
                pass
                
        elif action == len(self.hands[self.current_player]) + 1:
            # Pass action
            pass
        else:
            # Invalid action (should be caught by agent, but safe guard here)
            reward -= 10
            terminated = True # Penalize heavily and end episode
            
        # 2. Check Win Condition
        if self._check_win_condition():
            self.game_over = True
            terminated = True
            # Assign final rewards based on scores
            winner = np.argmax(self.scores)
            for p in range(self.num_players):
                if p == winner:
                    reward = 10 if p == self.current_player else 0
                else:
                    reward = -1 if p == self.current_player else 0
            info['winner'] = winner

        # 3. Switch Turn
        if not terminated:
            self.current_player = (self.current_player + 1) % self.num_players

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def _process_played_card(self, card, player_idx):
        """
        Implement specific game rules here.
        e.g., if card > last_played_card, score points.
        """
        # Placeholder: Just increment score for playing a high card
        if card > 20:
            self.scores[player_idx] += 1

    def _check_win_condition(self):
        """
        Return True if the game should end.
        e.g., Deck empty and hands empty, or someone reaches 10 points.
        """
        if len(self.deck) == 0 and all(len(h) == 0 for h in self.hands):
            return True
        if max(self.scores) >= 10:
            return True
        return False

    def _get_observation(self):
        """
        Returns a flattened numpy array representing the state.
        """
        obs = []
        
        # Encode Player Hand (Current Player)
        hand = self.hands[self.current_player]
        padded_hand = hand + [-1] * (self.max_hand_size - len(hand))
        obs.extend(padded_hand)
        
        # Encode Opponent Hand (Partial Observability: hide values, just show count or mask)
        # For full observability, uncomment the next line. 
        # For partial, we might just send the length or a mask.
        opp_hand = self.hands[(self.current_player + 1) % self.num_players]
        # Option A: Full info (not recommended for competitive games)
        # padded_opp = opp_hand + [-1] * (self.max_hand_size - len(opp_hand))
        # Option B: Partial info (only length or hidden)
        padded_opp = [-1] * self.max_hand_size # Hiding opponent cards
        
        obs.extend(padded_opp)
        
        # Deck count
        obs.append(len(self.deck))
        
        # Turn indicator (One-hot style or just current player ID)
        obs.append(self.current_player)
        
        return np.array(obs, dtype=np.int32)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n--- Turn: Player {self.current_player} ---")
            print(f"Deck: {len(self.deck)} cards")
            print(f"Player 0 Hand: {self.hands[0]}")
            print(f"Player 1 Hand: {self.hands[1]}")
            print(f"Scores: {self.scores}")