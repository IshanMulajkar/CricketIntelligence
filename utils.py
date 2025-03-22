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
    Get a list of upcoming IPL 2025 matches for the next 5 days
    In a real implementation, this would fetch from a cricket API
    
    Returns:
        list: List of dictionaries containing upcoming match details
    """
    # Generate upcoming IPL matches for the next 5 days
    matches = []
    
    # IPL teams with their home venues
    ipl_venues = {
        "Mumbai Indians": [
            {"name": "Wankhede Stadium", "city": "Mumbai", "pitch_type": "Batting friendly", "weather": "Clear"}
        ],
        "Chennai Super Kings": [
            {"name": "M. A. Chidambaram Stadium", "city": "Chennai", "pitch_type": "Spin friendly", "weather": "Clear"}
        ],
        "Royal Challengers Bangalore": [
            {"name": "M. Chinnaswamy Stadium", "city": "Bangalore", "pitch_type": "Batting friendly", "weather": "Clear"}
        ],
        "Kolkata Knight Riders": [
            {"name": "Eden Gardens", "city": "Kolkata", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Delhi Capitals": [
            {"name": "Arun Jaitley Stadium", "city": "Delhi", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Punjab Kings": [
            {"name": "IS Bindra Stadium", "city": "Mohali", "pitch_type": "Batting friendly", "weather": "Clear"}
        ],
        "Rajasthan Royals": [
            {"name": "Sawai Mansingh Stadium", "city": "Jaipur", "pitch_type": "Batting friendly", "weather": "Clear"}
        ],
        "Sunrisers Hyderabad": [
            {"name": "Rajiv Gandhi International Stadium", "city": "Hyderabad", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Gujarat Titans": [
            {"name": "Narendra Modi Stadium", "city": "Ahmedabad", "pitch_type": "Balanced", "weather": "Clear"}
        ],
        "Lucknow Super Giants": [
            {"name": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium", "city": "Lucknow", "pitch_type": "Balanced", "weather": "Clear"}
        ]
    }
    
    # IPL team strengths
    ipl_team_strengths = {
        "Mumbai Indians": {"batting": 9.0, "bowling": 8.5, "fielding": 8.5, "form": "Good"},
        "Chennai Super Kings": {"batting": 8.5, "bowling": 8.5, "fielding": 8.0, "form": "Good"},
        "Royal Challengers Bangalore": {"batting": 9.0, "bowling": 7.5, "fielding": 8.0, "form": "Average"},
        "Kolkata Knight Riders": {"batting": 8.0, "bowling": 8.0, "fielding": 8.5, "form": "Good"},
        "Delhi Capitals": {"batting": 8.0, "bowling": 8.0, "fielding": 8.0, "form": "Average"},
        "Punjab Kings": {"batting": 8.5, "bowling": 7.5, "fielding": 7.5, "form": "Poor"},
        "Rajasthan Royals": {"batting": 8.0, "bowling": 8.0, "fielding": 8.0, "form": "Average"},
        "Sunrisers Hyderabad": {"batting": 7.5, "bowling": 8.5, "fielding": 8.0, "form": "Good"},
        "Gujarat Titans": {"batting": 8.0, "bowling": 8.5, "fielding": 8.5, "form": "Average"},
        "Lucknow Super Giants": {"batting": 8.0, "bowling": 8.0, "fielding": 8.0, "form": "Average"}
    }
    
    # Generate realistic IPL match schedule
    today = datetime.now()
    
    # For IPL 2025 schedule (March-May 2025)
    # Assuming today is in the IPL season; otherwise, we'd set a specific start date
    
    # Predefined IPL matches for the next 5 days
    ipl_matches = [
        {
            'team1': "Mumbai Indians", 
            'team2': "Chennai Super Kings", 
            'date': today.strftime('%Y-%m-%d'),
            'venue': "Wankhede Stadium", 
            'city': "Mumbai",
            'match_type': "T20",
            'pitch_type': "Batting friendly",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 1
        },
        {
            'team1': "Royal Challengers Bangalore", 
            'team2': "Kolkata Knight Riders", 
            'date': today.strftime('%Y-%m-%d'),
            'venue': "M. Chinnaswamy Stadium", 
            'city': "Bangalore",
            'match_type': "T20",
            'pitch_type': "Batting friendly",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 2
        },
        {
            'team1': "Delhi Capitals", 
            'team2': "Gujarat Titans", 
            'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
            'venue': "Arun Jaitley Stadium", 
            'city': "Delhi",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 3
        },
        {
            'team1': "Punjab Kings", 
            'team2': "Lucknow Super Giants", 
            'date': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
            'venue': "IS Bindra Stadium", 
            'city': "Mohali",
            'match_type': "T20",
            'pitch_type': "Batting friendly",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 4
        },
        {
            'team1': "Sunrisers Hyderabad", 
            'team2': "Rajasthan Royals", 
            'date': (today + timedelta(days=2)).strftime('%Y-%m-%d'),
            'venue': "Rajiv Gandhi International Stadium", 
            'city': "Hyderabad",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 5
        },
        {
            'team1': "Chennai Super Kings", 
            'team2': "Royal Challengers Bangalore", 
            'date': (today + timedelta(days=2)).strftime('%Y-%m-%d'),
            'venue': "M. A. Chidambaram Stadium", 
            'city': "Chennai",
            'match_type': "T20",
            'pitch_type': "Spin friendly",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 6
        },
        {
            'team1': "Kolkata Knight Riders", 
            'team2': "Mumbai Indians", 
            'date': (today + timedelta(days=3)).strftime('%Y-%m-%d'),
            'venue': "Eden Gardens", 
            'city': "Kolkata",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Light rain",
            'tournament': "IPL 2025",
            'match_number': 7
        },
        {
            'team1': "Gujarat Titans", 
            'team2': "Sunrisers Hyderabad", 
            'date': (today + timedelta(days=3)).strftime('%Y-%m-%d'),
            'venue': "Narendra Modi Stadium", 
            'city': "Ahmedabad",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 8
        },
        {
            'team1': "Lucknow Super Giants", 
            'team2': "Rajasthan Royals", 
            'date': (today + timedelta(days=4)).strftime('%Y-%m-%d'),
            'venue': "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium", 
            'city': "Lucknow",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 9
        },
        {
            'team1': "Delhi Capitals", 
            'team2': "Punjab Kings", 
            'date': (today + timedelta(days=4)).strftime('%Y-%m-%d'),
            'venue': "Arun Jaitley Stadium", 
            'city': "Delhi",
            'match_type': "T20",
            'pitch_type': "Balanced",
            'weather': "Clear",
            'tournament': "IPL 2025",
            'match_number': 10
        }
    ]
    
    # Add special match conditions based on venue and weather
    for match in ipl_matches:
        # Adjust pitch conditions based on recent usage
        if match['match_number'] > 5:
            # Pitches might deteriorate as tournament progresses
            if match['pitch_type'] == "Batting friendly":
                match['pitch_type'] = "Balanced"
            elif match['pitch_type'] == "Balanced":
                match['pitch_type'] = "Spin friendly"
        
        # Add match time
        match_number = match['match_number']
        if match_number % 2 == 0:
            match['time'] = "19:30 IST"  # Evening match
        else:
            match['time'] = "15:30 IST"  # Afternoon match
        
        # Add team form and key players
        team1 = match['team1']
        team2 = match['team2']
        
        match['team1_form'] = ipl_team_strengths[team1]['form']
        match['team2_form'] = ipl_team_strengths[team2]['form']
        
        # Add current points for teams (simulated)
        match['team1_points'] = min(14, match['match_number'] // 2)
        match['team2_points'] = min(12, match['match_number'] // 3)
        
        # Add head-to-head stats
        head_to_head_wins = {
            'Mumbai Indians': {'Chennai Super Kings': 20, 'Royal Challengers Bangalore': 17},
            'Chennai Super Kings': {'Mumbai Indians': 15, 'Royal Challengers Bangalore': 19},
            'Royal Challengers Bangalore': {'Mumbai Indians': 12, 'Chennai Super Kings': 10},
            # Add more team combinations as needed
        }
        
        team1_wins = head_to_head_wins.get(team1, {}).get(team2, 5)
        team2_wins = head_to_head_wins.get(team2, {}).get(team1, 5)
        
        match['head_to_head'] = f"{team1} {team1_wins} - {team2_wins} {team2}"
    
    # Add all IPL matches
    matches.extend(ipl_matches)
    
    # Sort matches by date and match number
    matches.sort(key=lambda x: (x['date'], x.get('match_number', 0)))
    
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
