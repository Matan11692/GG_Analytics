

from Models.regex_extraction import RegexExtraction
import pandas as pd
import numpy as np
import uuid
import re
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime


def parse_tour_clean(log_text: str, normalize: bool = True) -> list[dict]:
    from RegexExtraction import RegexExtraction  

    hands = re.split(r'\n\s*\n', log_text.strip())
    hands = list(reversed([h for h in hands if h.strip()]))  

    parsed_hands = [RegexExtraction(h, normalize=normalize) for h in hands]
    all_rows = []

    for i in range(len(parsed_hands) - 1):
        current_parser = parsed_hands[i]

        try:
            general_data = {
                "Modality": current_parser.extract_modality(),
                "TableSize": current_parser.extract_table_size(),
                "BuyIn": current_parser.extract_buyin(),
                "TournID": current_parser.extract_tournament_id(),
                "HandID": current_parser.extract_hand_id(),
                "LocalTime": current_parser.extract_local_time(),
                "Level": current_parser.extract_level(),
                "Ante": current_parser.extract_ante(),
                "Blinds": current_parser.extract_blinds(),
                "BoardFlop": current_parser.extract_board_cards()[0],
                "BoardTurn": current_parser.extract_board_cards()[1],
                "BoardRiver": current_parser.extract_board_cards()[2],
                "HeroHand": current_parser.extract_hero_hand()
            }

            current_players = current_parser.extract_players_info()
            current_players = current_parser.sort_players_by_position(current_players)
            positions = current_parser.assign_positions(current_players)

            for player in current_players:
                name = player["Player"]
                current_stack = player["Stack"]

                row = {
                    **general_data,
                    "Playing": len(current_players),
                    "Player": name,
                    "Seat": player["Seat"],
                    "PostedAnte": current_parser.extract_posted_ante(name),
                    "PostedBlind": current_parser.extract_posted_blind(name),
                    "Position": positions.get(name),
                    "Stack": current_stack,
                    "PreflopAction": current_parser.extract_street_action("HOLE CARDS", name),
                    "FlopAction": current_parser.extract_street_action("FLOP", name),
                    "TurnAction": current_parser.extract_street_action("TURN", name),
                    "RiverAction": current_parser.extract_street_action("RIVER", name),
                    "AnteAllIn": current_parser.extract_ante_allin(name, current_players),
                    "PreflopAllIn": current_parser.extract_allin("HOLE CARDS", name),
                    "FlopAllIn": current_parser.extract_allin("FLOP", name),
                    "TurnAllIn": current_parser.extract_allin("TURN", name),
                    "RiverAllIn": current_parser.extract_allin("RIVER", name),
                    "ShowDown": current_parser.extract_showdown_cards(name),
                    "Result": current_parser.extract_result(name)
                }
                all_rows.append(row)

        except Exception as e:
            print(f"Error parsing hand: {e}")
            continue

    return all_rows




def parse_full_log_to_dataframe(log_text: str, game_type:str, normalize: bool = True) -> pd.DataFrame:
  if game_type == 'tour':
    return pd.DataFrame(parse_tour(log_text, normalize=normalize))
  elif game_type == 'cash':
    return pd.DataFrame(parse_cash(log_text, normalize=normalize))
  else:
    print('Game type is not exists')