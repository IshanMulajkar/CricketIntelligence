import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re

def format_team_name(team_name):
    """
    Format a team name for consistent display
    
    Args:
        team_name (str): Raw team name
        
    Returns:
        str: Formatted team name
    """
    if not team_name:
        return ""
    
    # Convert to title case
    formatted = team_name.title()
    
    # Handle special cases
    team_map = {
        "Newzealand": "New Zealand",
        "Srilanka": "Sri Lanka",
        "Southafrica": "South Africa",
        "Westindies": "West Indies",
        "Windies": "West Indies",
        "Uae": "UAE",
        "Usa": "USA",
        "Nz": "New Zealand",
        "Sa": "South Africa",
        "Wi": "West Indies"
    }
    
    return team_map.get(formatted, formatted)

def get_upcoming_matches():
    """
    Get a list of upcoming cricket matches
    In a real implementation, this would fetch from a cricket API
    
    Returns:
        list: List of dictionaries containing upcoming match details
    """
    # Generate upcoming matches for the next 14 days
    matches = []
    
    # Major cricket teams
    teams = [
        "India", "Australia", "England", "New Zealand", 
        "Pakistan", "South Africa", "West Indies", "Sri Lanka",
        "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland"
    ]
    
    # Venues
    venues = [
        "Melbourne Cricket Ground", "Lord's", "Eden Gardens", 
        "Wanderers Stadium", "Sydney Cricket Ground", 
        "Sharjah Cricket Stadium", "Kensington Oval",
        "R. Premadasa Stadium", "Dubai International Cricket Stadium"
    ]
    
    # Generate matches
    today = datetime.now()
    
    for i in range(14):
        # 50% chance of having a match on any given day
        if np.random.random() < 0.5:
            match_date = today + timedelta(days=i)
            
            # Randomly select two different teams
            match_teams = np.random.choice(teams, 2, replace=False)
            team1, team2 = match_teams
            
            # Randomly select venue
            venue = np.random.choice(venues)
            
            # Add to matches list
            matches.append({
                'team1': team1,
                'team2': team2,
                'date': match_date.strftime('%Y-%m-%d'),
                'venue': venue,
                'match_type': np.random.choice(["ODI", "T20I", "Test"], p=[0.4, 0.4, 0.2])
            })
    
    return matches

def extract_teams_from_query(query):
    """
    Extract team names from a user query
    
    Args:
        query (str): User query text
        
    Returns:
        tuple: Two team names if found, (None, None) otherwise
    """
    # Common team names and variations
    teams_map = {
        'india': ['india', 'indian', 'ind', 'men in blue'],
        'australia': ['australia', 'australian', 'aus', 'aussies'],
        'england': ['england', 'english', 'eng', 'three lions'],
        'new zealand': ['new zealand', 'nz', 'kiwis', 'black caps'],
        'pakistan': ['pakistan', 'pak', 'men in green'],
        'south africa': ['south africa', 'sa', 'proteas'],
        'west indies': ['west indies', 'wi', 'windies', 'caribbean'],
        'sri lanka': ['sri lanka', 'sl', 'lanka', 'lions'],
        'bangladesh': ['bangladesh', 'ban', 'bd', 'tigers'],
        'afghanistan': ['afghanistan', 'afg'],
        'zimbabwe': ['zimbabwe', 'zim', 'chevrons'],
        'ireland': ['ireland', 'ire']
    }
    
    # Convert query to lowercase
    query = query.lower()
    
    # Look for "X vs Y" pattern
    vs_pattern = re.compile(r'(\w+)\s+(?:vs|versus|and|v\.?s\.?|v)\s+(\w+)')
    match = vs_pattern.search(query)
    
    if match:
        team1_raw, team2_raw = match.groups()
        
        # Match to standard team names
        team1 = None
        team2 = None
        
        for std_name, variations in teams_map.items():
            # Check if team1_raw matches any variation
            if any(var in team1_raw or team1_raw in var for var in variations):
                team1 = std_name
            
            # Check if team2_raw matches any variation
            if any(var in team2_raw or team2_raw in var for var in variations):
                team2 = std_name
        
        if team1 and team2:
            return team1, team2
    
    # If "X vs Y" pattern not found, look for team names in the query
    found_teams = []
    
    for std_name, variations in teams_map.items():
        for var in variations:
            if f" {var} " in f" {query} ":
                found_teams.append(std_name)
                break
    
    # Remove duplicates while preserving order
    found_teams = list(dict.fromkeys(found_teams))
    
    if len(found_teams) >= 2:
        return found_teams[0], found_teams[1]
    
    return None, None

def calculate_implied_probability(odds):
    """
    Calculate implied probability from decimal odds
    
    Args:
        odds (float): Decimal odds
        
    Returns:
        float: Implied probability as a percentage
    """
    if odds <= 0:
        return 0
    
    return (1 / odds) * 100

def normalize_probabilities(probs):
    """
    Normalize a list of probabilities to sum to 100%
    
    Args:
        probs (list): List of probability values
        
    Returns:
        list: Normalized probabilities
    """
    total = sum(probs)
    
    if total <= 0:
        return probs
    
    return [p / total * 100 for p in probs]

def calculate_kelly_criterion(probability, odds, kelly_fraction=0.5):
    """
    Calculate Kelly Criterion stake for optimal betting
    
    Args:
        probability (float): Probability of winning (0-1)
        odds (float): Decimal odds
        kelly_fraction (float): Fraction of Kelly to use (0-1)
        
    Returns:
        float: Recommended stake as a percentage of bankroll
    """
    if odds <= 1 or probability <= 0 or probability >= 1:
        return 0
    
    q = 1 - probability
    b = odds - 1
    
    # Full Kelly formula: (bp - q) / b
    kelly = (b * probability - q) / b
    
    # Apply Kelly fraction (typically 0.25-0.5 to reduce variance)
    kelly = kelly * kelly_fraction
    
    # Ensure we don't get negative values
    return max(0, kelly * 100)
