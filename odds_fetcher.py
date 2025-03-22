import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time
from utils import get_upcoming_matches

def get_current_odds():
    """
    Fetch current betting odds for cricket matches
    In a real implementation, this would connect to a betting odds API
    
    Returns:
        list: List of dictionaries containing odds data for cricket matches
    """
    try:
        # In a real implementation, this would call an actual odds API
        # For now, we'll create simulated data
        
        # Create list of upcoming matches
        upcoming_matches = get_upcoming_matches()
        
        odds_data = []
        
        # Generate odds for each match
        for match in upcoming_matches:
            team1 = match['team1']
            team2 = match['team2']
            match_date = match['date']
            
            # Generate realistic odds
            # Lower odds (closer to 1) mean higher probability to win
            team1_odds = round(np.random.uniform(1.5, 4.0), 2)
            team2_odds = round(np.random.uniform(1.5, 4.0), 2)
            draw_odds = round(np.random.uniform(5.0, 10.0), 2)
            
            # Ensure probabilities don't exceed 100% (accounting for bookmaker margin)
            implied_prob = (1/team1_odds + 1/team2_odds + 1/draw_odds)
            if implied_prob > 1.1:  # Typical bookmaker margin is 5-10%
                factor = 1.1 / implied_prob
                team1_odds = round(team1_odds / factor, 2)
                team2_odds = round(team2_odds / factor, 2)
                draw_odds = round(draw_odds / factor, 2)
            
            # Add to odds data
            odds_data.append({
                'match': f"{team1} vs {team2}",
                'date': match_date,
                'team1': team1,
                'team1_odds': team1_odds,
                'team2': team2,
                'team2_odds': team2_odds,
                'draw_odds': draw_odds,
                'source': np.random.choice(['Bet365', 'Betfair', 'Ladbrokes', 'William Hill', 'Paddy Power']),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return odds_data
    
    except Exception as e:
        print(f"Error fetching odds data: {str(e)}")
        return []

def get_odds_by_match(team1, team2):
    """
    Get betting odds for a specific match
    
    Args:
        team1 (str): First team name
        team2 (str): Second team name
        
    Returns:
        dict: Dictionary containing odds data for the match
    """
    try:
        # Get all current odds
        all_odds = get_current_odds()
        
        # Find the match
        for match_odds in all_odds:
            if (match_odds['team1'].lower() == team1.lower() and match_odds['team2'].lower() == team2.lower()) or \
               (match_odds['team1'].lower() == team2.lower() and match_odds['team2'].lower() == team1.lower()):
                return match_odds
        
        # Match not found
        return None
    
    except Exception as e:
        print(f"Error fetching match odds: {str(e)}")
        return None

def get_best_odds(team):
    """
    Get the best available odds for a team across all their upcoming matches
    
    Args:
        team (str): Team name
        
    Returns:
        list: List of dictionaries containing best odds for the team
    """
    try:
        # Get all current odds
        all_odds = get_current_odds()
        
        # Find matches involving the team
        team_matches = []
        
        for match_odds in all_odds:
            if match_odds['team1'].lower() == team.lower():
                team_matches.append({
                    'match': match_odds['match'],
                    'opponent': match_odds['team2'],
                    'odds': match_odds['team1_odds'],
                    'date': match_odds['date'],
                    'source': match_odds['source']
                })
            elif match_odds['team2'].lower() == team.lower():
                team_matches.append({
                    'match': match_odds['match'],
                    'opponent': match_odds['team1'],
                    'odds': match_odds['team2_odds'],
                    'date': match_odds['date'],
                    'source': match_odds['source']
                })
        
        # Sort by date
        team_matches.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        return team_matches
    
    except Exception as e:
        print(f"Error fetching best odds: {str(e)}")
        return []

def compare_odds_sources(team1, team2):
    """
    Compare odds from different bookmakers for a specific match
    
    Args:
        team1 (str): First team name
        team2 (str): Second team name
        
    Returns:
        dict: Dictionary containing odds comparison from different sources
    """
    # In a real implementation, this would fetch odds from multiple sources
    # For now, we'll simulate this with random data
    
    sources = ['Bet365', 'Betfair', 'Ladbrokes', 'William Hill', 'Paddy Power']
    comparison = {}
    
    for source in sources:
        # Generate slightly different odds for each source
        team1_odds = round(np.random.uniform(1.8, 2.2), 2)
        team2_odds = round(np.random.uniform(1.8, 2.2), 2)
        draw_odds = round(np.random.uniform(7.0, 9.0), 2)
        
        comparison[source] = {
            'team1_odds': team1_odds,
            'team2_odds': team2_odds,
            'draw_odds': draw_odds
        }
    
    return comparison
