
from models.regex_extraction import RegexExtraction
import pandas as pd
import numpy as np
import uuid
import re
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
import re



def FilterAction(df: pd.DataFrame, column: str, action: str, street: int = None) -> pd.DataFrame:
    """
    Filter rows where the specified action appears in the given action column.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name like 'PreflopAction', 'FlopAction', etc.
        action (str): Action to look for: 'call', 'raise', 'fold', 'bet', etc.
        street (int, optional): 1-based index to match specific action; if None, scan all.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    def match(actions):
        if not isinstance(actions, list) or not actions:
            return False
        if street is None:
            return any(a[0] == action for a in actions if isinstance(a, list) and a[0] is not None)
        elif 1 <= street <= len(actions):
            a = actions[street - 1]
            return isinstance(a, list) and a[0] == action
        return False

    return df[df[column].apply(match)]
    



def FilterActionAmount(df: pd.DataFrame, column: str, comparison: str, amount: float, street: int = None) -> pd.DataFrame:
    """
    Filter rows based on comparison between action amount and a threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Action column like 'PreflopAction', 'FlopAction', etc.
        comparison (str): One of 'gte', 'lte', 'gt', 'lt', 'eq'
        amount (float): Value to compare against
        street (int, optional): 1-based index to filter a specific action in the list.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    def compare_value(val):
        if not isinstance(val, list) or not val:
            return False
        actions = [val[street - 1]] if street is not None and 1 <= street <= len(val) else val

        for action in actions:
            if not isinstance(action, list) or len(action) < 2:
                continue
            action_amt = action[1]
            if isinstance(action_amt, (int, float)):
                if comparison == 'gte' and action_amt >= amount:
                    return True
                elif comparison == 'lte' and action_amt <= amount:
                    return True
                elif comparison == 'gt' and action_amt > amount:
                    return True
                elif comparison == 'lt' and action_amt < amount:
                    return True
                elif comparison == 'eq' and action_amt == amount:
                    return True
        return False

    return df[df[column].apply(compare_value)]


def count_checks(actions):
    """
    Counts how many 'check' actions are present in the given action list.
    """
    if not isinstance(actions, list):
        return 0
    return sum(1 for action in actions if isinstance(action, list) and action[0] == 'check')


def is_active(actions):
    """
    Determines if the player was active on the street (i.e., performed any action other than fold/None).
    """
    if not isinstance(actions, list):
        return False
    return any(isinstance(action, list) and action[0] not in [None, 'fold'] for action in actions)


def detect_check_raises(df: pd.DataFrame, street: str) -> pd.DataFrame:
    """
    Filters hands where a player performed a check-raise on a given postflop street.

    Parameters:
    - df: DataFrame with hand data
    - street: One of 'Flop', 'Turn', 'River'

    Returns:
    - Filtered DataFrame containing only hands with a check-raise.
    """
    if street not in ['Flop', 'Turn', 'River']:
        raise ValueError("Street must be one of: 'Flop', 'Turn', or 'River'")

    street_col = f"{street}Action"
    check_raiser_col = f"{street}_CheckRaiser"

    def player_check_raised(actions):
        if not isinstance(actions, list):
            return False
        found_check = False
        for action in actions:
            if not isinstance(action, list) or not action:
                continue
            if action[0] == 'check':
                found_check = True
            elif found_check and action[0] in {'raise', 'bet'}:
                return True
        return False

    df = df.copy()
    df[check_raiser_col] = df[street_col].apply(player_check_raised)

    hands_with_cr = df[df[check_raiser_col]].HandID.unique()
    return df[df['HandID'].isin(hands_with_cr)]


def filter_postflop_players_by_position(
    df: pd.DataFrame,
    req_positions: List[str],
    number_of_players: int,
    street: str
) -> pd.DataFrame:
    """
    Filters hands where an exact number of players from given positions played postflop on the specified street.

    Parameters:
    - df: DataFrame of parsed hand histories.
    - req_positions: list of positions to consider (e.g., ['small blind', 'big blind']).
    - number_of_players: exact number of players from req_positions that must act postflop.
    - street: street to analyze ('Flop', 'Turn', 'River').
    """
    if street not in ['Flop', 'Turn', 'River']:
        raise ValueError("Street must be one of: 'Flop', 'Turn', or 'River'")

    if len(req_positions) < number_of_players:
        raise ValueError("Cannot require more players than positions provided")

    street_col = f"{street}Action"
    played_col = f"{street_col}_played"

    df_filtered = df[df['Position'].isin(req_positions)].copy()
    df_filtered[played_col] = df_filtered[street_col].apply(is_active)

    grouped = df_filtered.groupby('HandID')[played_col].sum().reset_index()
    valid_hand_ids = grouped[grouped[played_col] == number_of_players]['HandID'].tolist()

    return df[df['HandID'].isin(valid_hand_ids)]


def identify_single_raise_pot_preflop(group):
    raisers = set()
    callers = set()

    for _, row in group.iterrows():
        actions = row['PreflopAction']
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, list) and len(action) >= 2:
                if action[0] in ['raise', 'bet']:
                    raisers.add(row['Player'])
                elif action[0] == 'call':
                    callers.add(row['Player'])
        if len(raisers) > 1 or len(callers) > 1:
            return False

    return len(raisers) == 1 and len(callers) == 1


def get_preflop_aggresor(group):
    for _, row in group.iterrows():
        actions = row['PreflopAction']
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, list) and action[0] in ['raise', 'bet']:
                return row['Player']
    return None


def aggressor_bet_and_call_on_streets(
    group: pd.DataFrame,
    streets: List[str],
    get_at_least_one_call: bool = True
) -> bool:
    """
    Checks that the aggressor bet or raised and received at least one call
    on all specified street(s).

    Parameters:
    - group: DataFrame for one hand (i.e., grouped by HandID)
    - streets: list of streets to validate, e.g. ['Flop', 'River']
    - get_at_least_one_call: whether the aggressor needs to get a call

    Returns:
    - True if conditions met on all specified streets, else False
    """
    if group.empty or 'Aggressor' not in group.columns:
        return False

    aggressor = group['Aggressor'].iloc[0]

    if aggressor not in group['Player'].values:
        return False  # Aggressor not present in this hand data

    for street in streets:
        action_col = f"{street}Action"
        bet_by_agg = False
        call_by_other = False

        for _, row in group.iterrows():
            actions = row.get(action_col, [])
            if not isinstance(actions, list):
                continue

            for action in actions:
                if not isinstance(action, list) or len(action) < 2:
                    continue

                action_type = action[0]
                if row["Player"] == aggressor and action_type in ["bet", "raise"]:
                    bet_by_agg = True
                elif get_at_least_one_call and row["Player"] != aggressor and action_type == "call":
                    call_by_other = True

        # If we require a call and didn't get one → fail
        if not bet_by_agg or (get_at_least_one_call and not call_by_other):
            return False

    return True



# Third street - both players (check-check)
def filter_all_checked_on_street(df: pd.DataFrame, street: str) -> pd.DataFrame:
    """
    Filters hands where all active players checked on the given street.

    Parameters:
    - df: DataFrame with parsed hand histories.
    - street: One of 'Flop', 'Turn', or 'River'.

    Returns:
    - Filtered DataFrame where all active players checked on the street.
    """
    street = street.capitalize()
    street_col = f"{street}Action"
    check_col = f"{street}_Checks"
    active_col = f"{street}_Active"

    df = df.copy()
    df[check_col] = df[street_col].apply(count_checks)
    df[active_col] = df[street_col].apply(lambda x: 1 if is_active(x) else 0)

    grouped = df.groupby("HandID")[[check_col, active_col]].sum().reset_index()
    valid_hand_ids = grouped[grouped[check_col] == grouped[active_col]]['HandID'].tolist()

    return df[df['HandID'].isin(valid_hand_ids)]
