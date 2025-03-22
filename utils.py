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
    Get a list of upcoming cricket matches for the next 5 days
    In a real implementation, this would fetch from a cricket API
    
    Returns:
        list: List of dictionaries containing upcoming match details
    """
    # Generate upcoming matches for the next 5 days
    matches = []
    
    # Major cricket teams with their home venues
    team_home_venues = {
        "India": [
            {"name": "Eden Gardens", "city": "Kolkata", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Wankhede Stadium", "city": "Mumbai", "pitch_type": "Batting friendly", "weather": "Clear"},
            {"name": "M. Chinnaswamy Stadium", "city": "Bangalore", "pitch_type": "Batting friendly", "weather": "Clear"}
        ],
        "Australia": [
            {"name": "Melbourne Cricket Ground", "city": "Melbourne", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Sydney Cricket Ground", "city": "Sydney", "pitch_type": "Batting friendly", "weather": "Cloudy"},
            {"name": "Adelaide Oval", "city": "Adelaide", "pitch_type": "Bowling friendly", "weather": "Clear"}
        ],
        "England": [
            {"name": "Lord's", "city": "London", "pitch_type": "Bowling friendly", "weather": "Cloudy"},
            {"name": "The Oval", "city": "London", "pitch_type": "Batting friendly", "weather": "Light rain"},
            {"name": "Edgbaston", "city": "Birmingham", "pitch_type": "Balanced", "weather": "Cloudy"}
        ],
        "New Zealand": [
            {"name": "Eden Park", "city": "Auckland", "pitch_type": "Balanced", "weather": "Cloudy"},
            {"name": "Basin Reserve", "city": "Wellington", "pitch_type": "Bowling friendly", "weather": "Cloudy"},
            {"name": "Hagley Oval", "city": "Christchurch", "pitch_type": "Bowling friendly", "weather": "Clear"}
        ],
        "Pakistan": [
            {"name": "National Stadium", "city": "Karachi", "pitch_type": "Batting friendly", "weather": "Clear"},
            {"name": "Gaddafi Stadium", "city": "Lahore", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Rawalpindi Cricket Stadium", "city": "Rawalpindi", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "South Africa": [
            {"name": "Wanderers Stadium", "city": "Johannesburg", "pitch_type": "Batting friendly", "weather": "Clear"},
            {"name": "Newlands", "city": "Cape Town", "pitch_type": "Bowling friendly", "weather": "Clear"},
            {"name": "SuperSport Park", "city": "Centurion", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "West Indies": [
            {"name": "Kensington Oval", "city": "Bridgetown", "pitch_type": "Batting friendly", "weather": "Clear"},
            {"name": "Sabina Park", "city": "Kingston", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Queen's Park Oval", "city": "Port of Spain", "pitch_type": "Balanced", "weather": "Light rain"}
        ],
        "Sri Lanka": [
            {"name": "R. Premadasa Stadium", "city": "Colombo", "pitch_type": "Spin friendly", "weather": "Light rain"},
            {"name": "Galle International Stadium", "city": "Galle", "pitch_type": "Spin friendly", "weather": "Clear"},
            {"name": "Pallekele International Cricket Stadium", "city": "Kandy", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Bangladesh": [
            {"name": "Shere Bangla National Stadium", "city": "Dhaka", "pitch_type": "Spin friendly", "weather": "Clear"},
            {"name": "Zahur Ahmed Chowdhury Stadium", "city": "Chittagong", "pitch_type": "Spin friendly", "weather": "Clear"},
            {"name": "Sylhet International Cricket Stadium", "city": "Sylhet", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Afghanistan": [
            {"name": "Sharjah Cricket Stadium", "city": "Sharjah", "pitch_type": "Batting friendly", "weather": "Clear"},
            {"name": "Dubai International Cricket Stadium", "city": "Dubai", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Greater Noida Sports Complex Ground", "city": "Greater Noida", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Zimbabwe": [
            {"name": "Harare Sports Club", "city": "Harare", "pitch_type": "Balanced", "weather": "Clear"},
            {"name": "Queens Sports Club", "city": "Bulawayo", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Ireland": [
            {"name": "Malahide Cricket Club", "city": "Dublin", "pitch_type": "Bowling friendly", "weather": "Light rain"},
            {"name": "Clontarf Cricket Club", "city": "Dublin", "pitch_type": "Bowling friendly", "weather": "Cloudy"}
        ]
    }
    
    # Generate realistic matches
    today = datetime.now()
    
    # Predefined matches for the next 5 days
    predefined_matches = [
        {
            'team1': "India", 
            'team2': "Bangladesh", 
            'date': today.strftime('%Y-%m-%d'),
            'venue': "Eden Gardens", 
            'city': "Kolkata",
            'match_type': "ODI",
            'pitch_type': "Balanced",
            'weather': "Clear"
        },
        {
            'team1': "Australia", 
            'team2': "England", 
            'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
            'venue': "Melbourne Cricket Ground", 
            'city': "Melbourne",
            'match_type': "Test",
            'pitch_type': "Balanced",
            'weather': "Clear"
        },
        {
            'team1': "South Africa", 
            'team2': "New Zealand", 
            'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
            'venue': "SuperSport Park", 
            'city': "Centurion",
            'match_type': "T20I",
            'pitch_type': "Balanced",
            'weather': "Clear"
        },
        {
            'team1': "Pakistan", 
            'team2': "Sri Lanka", 
            'date': (today + timedelta(days=2)).strftime('%Y-%m-%d'),
            'venue': "National Stadium", 
            'city': "Karachi",
            'match_type': "ODI",
            'pitch_type': "Batting friendly",
            'weather': "Clear"
        },
        {
            'team1': "West Indies", 
            'team2': "Afghanistan", 
            'date': (today + timedelta(days=2)).strftime('%Y-%m-%d'),
            'venue': "Kensington Oval", 
            'city': "Bridgetown",
            'match_type': "T20I",
            'pitch_type': "Batting friendly",
            'weather': "Clear"
        },
        {
            'team1': "Zimbabwe", 
            'team2': "Ireland", 
            'date': (today + timedelta(days=3)).strftime('%Y-%m-%d'),
            'venue': "Harare Sports Club", 
            'city': "Harare",
            'match_type': "ODI",
            'pitch_type': "Balanced",
            'weather': "Clear"
        },
        {
            'team1': "Bangladesh", 
            'team2': "New Zealand", 
            'date': (today + timedelta(days=4)).strftime('%Y-%m-%d'),
            'venue': "Shere Bangla National Stadium", 
            'city': "Dhaka",
            'match_type': "Test",
            'pitch_type': "Spin friendly",
            'weather': "Clear"
        },
        {
            'team1': "India", 
            'team2': "Australia", 
            'date': (today + timedelta(days=5)).strftime('%Y-%m-%d'),
            'venue': "Wankhede Stadium", 
            'city': "Mumbai",
            'match_type': "ODI",
            'pitch_type': "Batting friendly",
            'weather': "Clear"
        }
    ]
    
    # Add all predefined matches
    matches.extend(predefined_matches)
    
    # Sort matches by date
    matches.sort(key=lambda x: x['date'])
    
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
