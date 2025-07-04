from models.regex_extraction import RegexExtraction
import pandas as pd
import numpy as np
import uuid
import re
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
import re



class RegexExtraction:
    def __init__(self, hand_text: str,normalize: bool = True):
        self.hand_text = hand_text
        self.header = hand_text.splitlines()[0]
        self.normalize = normalize
        self._blinds = self.extract_blinds()


    # ----------- CALCS -----------
    def normalize_amount(self, amount: Optional[float]) -> Optional[float]:
        if not self.normalize or amount is None:
            return amount
        bb = self._blinds[1]
        return round(amount / bb, 2) if bb else amount

    def extract_blinds(self) -> List[float]:
        # Tournament style (e.g. (150/300))
        match = re.search(r'\(([\d,]+)/([\d,]+)\)', self.header)
        if match:
            return [
                float(match.group(1).replace(",", "")),
                float(match.group(2).replace(",", ""))
            ]

        # Cash game style (e.g. ($2/$4))
        match_cash = re.search(r'\(\$([\d,.]+)\/\$([\d,.]+)\)', self.header)
        if match_cash:
            return [
                float(match_cash.group(1).replace(",", "")),
                float(match_cash.group(2).replace(",", ""))
            ]

        return [None, None]




    # ----------- HEADER / GENERAL INFO -----------
    def extract_modality(self) -> str:
        match = re.search(r'Tournament #\d+, (.+?) - Level', self.header)
        return match.group(1).strip() if match else None

    def extract_table_size(self) -> str:
        match = re.search(r'Table \'\' (.+?) Seat #\d+ is the button', self.hand_text)
        return match.group(1).strip() if match else None

    def extract_buyin(self) -> List[int]:
        match = re.search(r'\((\d+)\+(\d+)\+\d+\)', self.header)
        return [int(match.group(1)), int(match.group(2))] if match else [None, None]

    def extract_tournament_id(self) -> str:
        match = re.search(r'Tournament (#\d+)', self.header)
        return match.group(1) if match else None

    def extract_hand_id(self) -> str:
        match = re.search(r'Poker Hand #tour_(\d+)', self.header)
        return match.group(1) if match else None

    def extract_local_time(self) -> Optional[datetime]:
        match = re.search(r' - (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', self.header)
        return datetime.strptime(match.group(1), '%Y/%m/%d %H:%M:%S') if match else None

    def extract_level(self) -> str:
        match = re.search(r'-( Level\d+)', self.header)
        return match.group(1) if match else None

    def extract_ante(self) -> float:
        match = re.search(r'posts the ante (\d{1,3}(?:,\d{3})*)', self.hand_text)
        return float(match.group(1).replace(",", "")) if match else None

    def extract_blinds(self) -> List[float]:
        match = re.search(r'\(([\d,]+)/([\d,]+)\)', self.header)
        if match:
            return [float(match.group(1).replace(",", "")), float(match.group(2).replace(",", ""))]
        return [None, None]

    # ----------- SEATING AND PLAYER INFO -----------
    def extract_players_info(self) -> List[Dict]:
        players = []
        for seat_match in re.finditer(r'Seat (\d+): ([\w\d]+) \(([\d,]+) in chips\)', self.hand_text):
            seat, name, stack = seat_match.groups()
            stack_val = float(stack.replace(",", ""))
            players.append({
                "Seat": int(seat),
                "Player": name,
                "Stack": self.normalize_amount(stack_val),
            })
        return players

    def extract_hero_hand(self) -> List[str]:
        match = re.search(r'Dealt to Hero \[(.*?)\]', self.hand_text)
        return match.group(1).split() if match else []

    def extract_posted_ante(self, player: str) -> float:
        match = re.search(rf'{player}:\s+posts the ante ([\d,]+)', self.hand_text)
        return float(match.group(1).replace(",", "")) if match else 0.0

    def extract_posted_blind(self, player: str) -> float:
        match = re.search(rf'{player}:\s+posts (small|big) blind ([\d,]+)', self.hand_text)
        return float(match.group(2).replace(",", "")) if match else 0.0

    # ----------- POSITION -----------
    @staticmethod
    def get_positions_order(num_players: int) -> List[str]:
        if num_players == 2:
            return ["small blind", "button"]
        elif num_players == 3:
            return ["small blind", "big blind", "button"]
        elif num_players == 4:
            return ["small blind", "big blind", "UTG", "button"]
        elif num_players == 5:
            return ["small blind", "big blind", "UTG", "CO", "button"]
        elif num_players == 6:
            return ["small blind", "big blind", "UTG", "UTG1", "CO", "button"]
        elif num_players == 7:
            return ["small blind", "big blind", "UTG", "UTG1", "HJ", "CO", "button"]
        elif num_players == 8:
            return ["small blind", "big blind", "UTG", "UTG1", "MP", "HJ", "CO", "button"]
        elif num_players == 9:
            return ["small blind", "big blind", "UTG", "UTG1", "MP", "MP1", "HJ", "CO", "button"]
        else:
            return [f"Seat {i}" for i in range(num_players)]

    def assign_positions(self, players: List[Dict]) -> Dict[str, str]:
        # Extract player order from ante posting
        ante_order = re.findall(r'(\w+): posts the ante', self.hand_text)
        position_labels = self.get_positions_order(len(ante_order))

        return {player: pos for player, pos in zip(ante_order, position_labels)}


    def sort_players_by_position(self, players: List[Dict]) -> List[Dict]:
        position_map = self.assign_positions(players)
        for p in players:
            p["Position"] = position_map.get(p["Player"], f"Seat {p['Seat']}")

        ordered_roles = self.get_positions_order(len(players))

        # Only sort players with valid positions from the ordered_roles list
        def safe_index(pos):
            try:
                return ordered_roles.index(pos)
            except ValueError:
                return len(ordered_roles)  # push unknowns to the end

        players_sorted = sorted(players, key=lambda p: safe_index(p["Position"]))
        return players_sorted



    # ----------- ACTIONS & ALL-INS -----------
    def extract_street_action(self, street: str, player: str) -> List[List[str]]:
        street_map = {
            "preflop": "HOLE CARDS",
            "flop": "FLOP",
            "turn": "TURN",
            "river": "RIVER"
        }
        street_key = street_map.get(street.lower(), street.upper())
        pattern = rf'\*\*\* {street_key} \*\*\*(.*?)(?=\*\*\*|$)'
        match = re.search(pattern, self.hand_text, re.DOTALL)
        if not match:
            return [[None]]

        street_text = match.group(1).strip()
        actions = []

        for line in street_text.splitlines():
            if not line.startswith(player + ":"):
                continue
            content = line[len(player)+1:].strip().lower()

            if re.match(r'[\d,]+\s+to\s+[\d,]+', content):
                try:
                    final_amount = float(re.search(r'to\s+([\d,]+)', content).group(1).replace(",", ""))
                    actions.append(["bet", self.normalize_amount(final_amount)])
                    continue
                except:
                    actions.append(["bet", None])
                    continue

            if "raises" in content and "to" in content:
                try:
                    final_amount = float(re.search(r'to\s+([\d,]+)', content).group(1).replace(",", ""))
                    actions.append(["raise", self.normalize_amount(final_amount)])
                except:
                    actions.append(["raise", None])
            elif "bets" in content:
                try:
                    amount = float(re.search(r'bets\s+([\d,]+)', content).group(1).replace(",", ""))
                    actions.append(["bet", self.normalize_amount(amount)])
                except:
                    actions.append(["bet", None])
            elif "calls" in content:
                try:
                    amount = float(re.search(r'calls\s+([\d,]+)', content).group(1).replace(",", ""))
                    actions.append(["call", self.normalize_amount(amount)])
                except:
                    actions.append(["call", None])
            elif "folds" in content:
                actions.append(["fold", None])
            elif "checks" in content:
                actions.append(["check", None])
            else:
                actions.append([content.split()[0], None])

        return actions or [[None]]



    def extract_allin(self, street: str, player: str) -> bool:
        pattern = rf'\*\*\* {street.upper()} \*\*\*(.*?)(?=\*\*\*|$)'
        match = re.search(pattern, self.hand_text, re.DOTALL)
        return bool(re.search(rf'{player}:.*all-in', match.group(1))) if match else False

    def extract_ante_allin(self, player: str, players: List[Dict]) -> bool:
      """
      Determine if a player went all-in during ante posting.
      This happens when the ante posted is equal to or greater than their full stack.
      """
      posted_ante = self.extract_posted_ante(player)
      stack = next((p["Stack"] for p in players if p["Player"] == player), 0.0)

      return posted_ante >= stack and posted_ante > 0

    # ----------- BOARD CARDS -----------
    def extract_board_cards(self) -> Tuple[List[str], List[str], List[str]]:
        flop, turn, river = [], [], []
        flop_match = re.search(r'\*\*\* FLOP \*\*\* \[(.*?)\]', self.hand_text)
        turn_match = re.search(r'\*\*\* TURN \*\*\* \[.*?\] \[(.*?)\]', self.hand_text)
        river_match = re.search(r'\*\*\* RIVER \*\*\* \[.*?\] \[(.*?)\]', self.hand_text)

        if flop_match: flop = flop_match.group(1).split()
        if turn_match: turn = flop + [turn_match.group(1)]
        if river_match: river = turn + [river_match.group(1)]

        return flop, turn, river

    # ----------- RESULT & WINNINGS -----------
    def extract_showdown_cards(self, player: str) -> List[str]:
        match = re.search(rf'{player}: shows \[(.*?)\]', self.hand_text)
        return match.group(1).split() if match else []


    def extract_result(self, player: str) -> str:
        summary = self.hand_text.split("*** SUMMARY ***")[-1]
        showdown_section = re.search(r"\*\*\* SHOWDOWN \*\*\*(.*?)(?=\*\*\*|\Z)", self.hand_text, re.DOTALL)
        showdown_text = showdown_section.group(1) if showdown_section else ""
        river_exists = "*** RIVER ***" in self.hand_text
        turn_exists = "*** TURN ***" in self.hand_text
        flop_exists = "*** FLOP ***" in self.hand_text

        # Capitalized results from summary with optional (role)
        if re.search(rf"{player}(?:\(\w+ blind\))?\s+folded before Flop", summary):
            return "Folded Pre Flop"
        elif re.search(rf"{player}(?:\(\w+ blind\))?\s+folded on the Flop", summary):
            return "Folded On The Flop"
        elif re.search(rf"{player}(?:\(\w+ blind\))?\s+folded on the Turn", summary):
            return "Folded On The Turn"
        elif re.search(rf"{player}(?:\(\w+ blind\))?\s+folded on the River", summary):
            return "Folded On The River"

        # Won categories
        if f"{player} collected" in self.hand_text and len(set(re.findall(r'(\w+)\s+collected\s+[\d,]+', showdown_text))) == 1:
            if not flop_exists:
                return "Won Pre Flop"
            elif flop_exists and not turn_exists:
                return "Won At The Flop"
            elif turn_exists and not river_exists:
                return "Won At The Turn"
            elif river_exists and "Uncalled bet" in self.hand_text and player in self.hand_text.split("Uncalled bet")[-1]:
                return "Won At The River"
            else:
                return "Won At Showdown"

        # Split logic
        collected_matches = re.findall(r'(\w+)\s+collected\s+([\d,]+)', showdown_text)
        collected_dict = {}
        for p, amt in collected_matches:
            amt = float(amt.replace(",", ""))
            collected_dict.setdefault(p, []).append(amt)

        player_collected = collected_dict.get(player, [])
        all_totals = {p: sum(v) for p, v in collected_dict.items()}
        player_total = sum(player_collected)
        totals = list(all_totals.values())

        if player_collected:
            if len(all_totals) == 1:
                return "Won At Showdown"
            if totals.count(player_total) > 1:
                return "Split"
            elif player_total == min(totals):
                return "Split Main Pot"
            elif player_total == max(totals):
                return "Split Secondary Pot"




        # Eliminated: went all-in and didn’t win
        is_allin = (
            self.extract_ante_allin(player, self.extract_players_info()) or
            self.extract_allin("HOLE CARDS", player) or
            self.extract_allin("FLOP", player) or
            self.extract_allin("TURN", player) or
            self.extract_allin("RIVER", player)
        )

        if (
            river_exists and
            f"{player} collected" not in showdown_text and
            not is_allin and
            player not in showdown_text and
            f"{player}" in self.hand_text):  # appeared at some point in the hand

            return "Lost At Showdown"


        if is_allin:
            if self.extract_ante_allin(player, self.extract_players_info()) or self.extract_allin("HOLE CARDS", player):
                return "Eliminated Pre Flop"
            elif self.extract_allin("FLOP", player):
                return "Eliminated On The Flop"
            elif self.extract_allin("TURN", player):
                return "Eliminated On The Turn"
            elif self.extract_allin("RIVER", player):
                return "Eliminated On The River"

        return "Lost"

    # THIS IS A GOOD VERSION BUT NOT PERFECT, I STILL HAVE SOME BUGS HERE. THE VALIDATION IS CHECK ALL THE VALUES SUMING UP TO
    # df[['HandID','Balance']].groupby('HandID').sum().reset_index()['Balance'].value_counts()
    def extract_balance(self, player: str) -> float:
        try:
            balance = 0.0

            ante = self.extract_posted_ante(player)
            if ante:
                balance -= ante

            blind = self.extract_posted_blind(player)
            if blind:
                balance -= blind

            for street in ["preflop", "flop", "turn", "river"]:
                actions = self.extract_street_action(street, player)
                for action in actions:
                    if (
                        action
                        and isinstance(action, list)
                        and action[0] in {"bet", "raise", "call"}
                        and isinstance(action[1], (int, float))
                    ):
                        balance -= action[1]

            collected_matches = re.findall(rf'{player} collected ([\d,]+)', self.hand_text)
            for match in collected_matches:
                amount = float(match.replace(",", ""))
                balance += self.normalize_amount(amount)

            returned_matches = re.findall(rf'Uncalled bet \(([\d,]+)\) returned to {player}', self.hand_text)
            for match in returned_matches:
                amount = float(match.replace(",", ""))
                balance += self.normalize_amount(amount)

            return round(balance, 2)

        except Exception as e:
            print(f"Balance calc error for {player}: {e}")
            return 0.0

