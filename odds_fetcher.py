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
        
        # Team strength ratings (higher is better, scale of 1-10)
        team_ratings = {
            "India": 9.2,
            "Australia": 9.0,
            "England": 8.8,
            "New Zealand": 8.5,
            "Pakistan": 8.2,
            "South Africa": 8.1,
            "West Indies": 7.7,
            "Sri Lanka": 7.6,
            "Bangladesh": 7.2,
            "Afghanistan": 7.0,
            "Zimbabwe": 6.5,
            "Ireland": 6.2
        }
        
        # Format specific advantages (how much teams excel in different formats, scale 0.8-1.2)
        format_advantages = {
            "India": {"Test": 1.1, "ODI": 1.1, "T20I": 1.05},
            "Australia": {"Test": 1.15, "ODI": 1.05, "T20I": 1.0},
            "England": {"Test": 1.05, "ODI": 1.1, "T20I": 1.1},
            "New Zealand": {"Test": 1.1, "ODI": 1.05, "T20I": 1.0},
            "Pakistan": {"Test": 0.95, "ODI": 1.0, "T20I": 1.1},
            "South Africa": {"Test": 1.05, "ODI": 1.0, "T20I": 1.0},
            "West Indies": {"Test": 0.9, "ODI": 0.95, "T20I": 1.15},
            "Sri Lanka": {"Test": 1.0, "ODI": 0.95, "T20I": 0.95},
            "Bangladesh": {"Test": 0.9, "ODI": 0.95, "T20I": 0.95},
            "Afghanistan": {"Test": 0.85, "ODI": 0.9, "T20I": 1.05},
            "Zimbabwe": {"Test": 0.85, "ODI": 0.9, "T20I": 0.9},
            "Ireland": {"Test": 0.8, "ODI": 0.85, "T20I": 0.9}
        }
        
        # Weather impact on match outcomes (modifiers for different conditions)
        weather_impact = {
            "Clear": {"advantage": None, "draw_factor": 0.8},  # No advantage to either team, less chance of draw
            "Cloudy": {"advantage": "bowling", "draw_factor": 1.0},  # Slight bowling advantage
            "Light rain": {"advantage": "bowling", "draw_factor": 1.5},  # Bowling advantage, higher draw chance
            "Heavy rain": {"advantage": None, "draw_factor": 3.0}  # Much higher draw chance
        }
        
        # Pitch type impact
        pitch_impact = {
            "Batting friendly": {"advantage": "batting", "draw_factor": 0.9},
            "Bowling friendly": {"advantage": "bowling", "draw_factor": 1.1},
            "Balanced": {"advantage": None, "draw_factor": 1.0},
            "Spin friendly": {"advantage": "spin", "draw_factor": 1.0}
        }
        
        # Team strengths in different conditions
        team_conditions = {
            "India": {"batting": 9.5, "bowling": 8.9, "spin": 9.5},
            "Australia": {"batting": 9.0, "bowling": 9.2, "spin": 7.5},
            "England": {"batting": 9.0, "bowling": 8.7, "spin": 7.8},
            "New Zealand": {"batting": 8.3, "bowling": 8.7, "spin": 7.5},
            "Pakistan": {"batting": 8.0, "bowling": 8.5, "spin": 8.2},
            "South Africa": {"batting": 8.2, "bowling": 8.4, "spin": 7.0},
            "West Indies": {"batting": 8.0, "bowling": 7.5, "spin": 6.5},
            "Sri Lanka": {"batting": 7.8, "bowling": 7.5, "spin": 8.5},
            "Bangladesh": {"batting": 7.2, "bowling": 7.0, "spin": 8.5},
            "Afghanistan": {"batting": 6.8, "bowling": 7.0, "spin": 8.2},
            "Zimbabwe": {"batting": 6.5, "bowling": 6.5, "spin": 6.0},
            "Ireland": {"batting": 6.3, "bowling": 6.2, "spin": 5.5}
        }
        
        # Generate odds for each match based on team strengths and conditions
        for match in upcoming_matches:
            team1 = match['team1']
            team2 = match['team2']
            match_date = match['date']
            match_type = match.get('match_type', 'ODI')
            venue = match.get('venue', 'Neutral')
            weather = match.get('weather', 'Clear')
            pitch_type = match.get('pitch_type', 'Balanced')
            
            # Base strength calculation
            team1_strength = team_ratings.get(team1, 7.0)
            team2_strength = team_ratings.get(team2, 7.0)
            
            # Apply format-specific advantages
            team1_format_advantage = format_advantages.get(team1, {"Test": 1.0, "ODI": 1.0, "T20I": 1.0}).get(match_type, 1.0)
            team2_format_advantage = format_advantages.get(team2, {"Test": 1.0, "ODI": 1.0, "T20I": 1.0}).get(match_type, 1.0)
            
            team1_strength *= team1_format_advantage
            team2_strength *= team2_format_advantage
            
            # Apply home ground advantage (5-10% boost)
            if "India" in venue and team1 == "India":
                team1_strength *= 1.1
            elif "Australia" in venue and team1 == "Australia":
                team1_strength *= 1.1
            elif "Lord's" in venue and team1 == "England":
                team1_strength *= 1.1
            
            if "India" in venue and team2 == "India":
                team2_strength *= 1.1
            elif "Australia" in venue and team2 == "Australia":
                team2_strength *= 1.1
            elif "Lord's" in venue and team2 == "England":
                team2_strength *= 1.1
            
            # Apply weather and pitch conditions
            weather_advantage = weather_impact.get(weather, {"advantage": None, "draw_factor": 1.0}).get("advantage")
            pitch_advantage = pitch_impact.get(pitch_type, {"advantage": None, "draw_factor": 1.0}).get("advantage")
            
            if weather_advantage == "bowling":
                team1_bowling = team_conditions.get(team1, {"bowling": 7.0}).get("bowling", 7.0)
                team2_bowling = team_conditions.get(team2, {"bowling": 7.0}).get("bowling", 7.0)
                if team1_bowling > team2_bowling:
                    team1_strength *= 1.05
                else:
                    team2_strength *= 1.05
            
            if pitch_advantage == "batting":
                team1_batting = team_conditions.get(team1, {"batting": 7.0}).get("batting", 7.0)
                team2_batting = team_conditions.get(team2, {"batting": 7.0}).get("batting", 7.0)
                if team1_batting > team2_batting:
                    team1_strength *= 1.05
                else:
                    team2_strength *= 1.05
            elif pitch_advantage == "spin":
                team1_spin = team_conditions.get(team1, {"spin": 7.0}).get("spin", 7.0)
                team2_spin = team_conditions.get(team2, {"spin": 7.0}).get("spin", 7.0)
                if team1_spin > team2_spin:
                    team1_strength *= 1.08
                else:
                    team2_strength *= 1.08
            
            # Calculate draw probability factor based on match type, weather and pitch
            base_draw_factor = 1.0
            if match_type == "Test":
                base_draw_factor = 2.0  # Tests have higher draw probability
            elif match_type == "ODI":
                base_draw_factor = 0.5  # ODIs have lower draw probability
            elif match_type == "T20I":
                base_draw_factor = 0.3  # T20Is have even lower draw probability
                
            weather_draw_factor = weather_impact.get(weather, {"draw_factor": 1.0}).get("draw_factor", 1.0)
            pitch_draw_factor = pitch_impact.get(pitch_type, {"draw_factor": 1.0}).get("draw_factor", 1.0)
            
            # Adjust for team evenness (closer teams = higher draw chance)
            team_diff = abs(team1_strength - team2_strength)
            evenness_factor = 1.0 - (team_diff / 5.0)  # Scale of 0.0 to 1.0
            
            # Final draw factor
            draw_factor = base_draw_factor * weather_draw_factor * pitch_draw_factor * evenness_factor
            
            # Convert strengths to probabilities
            total_strength = team1_strength + team2_strength
            team1_prob = team1_strength / total_strength
            team2_prob = team2_strength / total_strength
            
            # Calculate draw probability (higher for Tests, affected by weather)
            draw_prob = 0.05 * draw_factor  # Base 5% chance adjusted by factors
            
            # Normalize probabilities
            total_prob = team1_prob + team2_prob + draw_prob
            team1_prob = team1_prob / total_prob
            team2_prob = team2_prob / total_prob
            draw_prob = draw_prob / total_prob
            
            # Convert to odds (1/probability) with bookmaker margin
            margin = np.random.uniform(1.05, 1.10)  # 5-10% margin
            
            team1_odds = round((1 / team1_prob) * margin, 2)
            team2_odds = round((1 / team2_prob) * margin, 2)
            draw_odds = round((1 / draw_prob) * margin, 2)
            
            # Add slight randomness to make it more realistic
            team1_odds = round(team1_odds * np.random.uniform(0.95, 1.05), 2)
            team2_odds = round(team2_odds * np.random.uniform(0.95, 1.05), 2)
            draw_odds = round(draw_odds * np.random.uniform(0.95, 1.05), 2)
            
            # Ensure minimum odds values
            team1_odds = max(1.01, team1_odds)
            team2_odds = max(1.01, team2_odds)
            draw_odds = max(1.01, draw_odds)
            
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
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'venue': match.get('venue', 'Neutral'),
                'city': match.get('city', ''),
                'match_type': match_type,
                'weather': weather,
                'pitch_type': pitch_type
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
    # Get the baseline odds first
    match_odds = get_odds_by_match(team1, team2)
    
    # Default odds in case no match is found
    default_odds = {
        'team1': team1,
        'team2': team2,
        'team1_odds': 2.0,
        'team2_odds': 2.0,
        'draw_odds': 5.0
    }
    
    if not match_odds:
        # Match not found, try to find the upcoming match
        upcoming_matches = get_upcoming_matches()
        match_found = False
        
        for match in upcoming_matches:
            if (match['team1'] == team1 and match['team2'] == team2) or \
               (match['team1'] == team2 and match['team2'] == team1):
                match_found = True
                # Match found in upcoming matches but not in odds data
                # Simulation will be based on team strength ratings
                
                # Team strength ratings (higher is better, scale of 1-10)
                team_ratings = {
                    "India": 9.2,
                    "Australia": 9.0,
                    "England": 8.8,
                    "New Zealand": 8.5,
                    "Pakistan": 8.2,
                    "South Africa": 8.1,
                    "West Indies": 7.7,
                    "Sri Lanka": 7.6,
                    "Bangladesh": 7.2,
                    "Afghanistan": 7.0,
                    "Zimbabwe": 6.5,
                    "Ireland": 6.2
                }
                
                # Calculate base odds based on team ratings
                team1_rating = team_ratings.get(team1, 7.0)
                team2_rating = team_ratings.get(team2, 7.0)
                total_rating = team1_rating + team2_rating
                
                team1_base_prob = team1_rating / total_rating
                team2_base_prob = team2_rating / total_rating
                
                # Base odds with 5% draw probability
                draw_prob = 0.05
                team1_prob = team1_base_prob * (1 - draw_prob)
                team2_prob = team2_base_prob * (1 - draw_prob)
                
                team1_base_odds = round(1 / team1_prob, 2)
                team2_base_odds = round(1 / team2_prob, 2)
                draw_base_odds = round(1 / draw_prob, 2)
                
                # Create base match odds to use as reference
                match_odds = {
                    'team1': team1,
                    'team2': team2,
                    'team1_odds': team1_base_odds,
                    'team2_odds': team2_base_odds,
                    'draw_odds': draw_base_odds
                }
                break
        
        # If no match was found in upcoming matches, use default odds
        if not match_found:
            match_odds = default_odds
    
    sources = ['Bet365', 'Betfair', 'Ladbrokes', 'William Hill', 'Paddy Power']
    comparison = {}
    
    # Each bookmaker has its own characteristics
    bookmaker_profiles = {
        'Bet365': {
            'margin': 1.07,  # 7% margin
            'variance': 0.03,  # Relatively consistent odds
            'bias': {
                'favorites': 0.98,  # Slightly lower odds on favorites
                'underdogs': 1.02,  # Slightly higher odds on underdogs
                'draws': 1.00      # Neutral on draws
            }
        },
        'Betfair': {
            'margin': 1.05,  # Lower margin (exchange)
            'variance': 0.04,  # More variance in odds
            'bias': {
                'favorites': 1.00,  # Neutral on favorites
                'underdogs': 1.01,  # Slightly higher odds on underdogs
                'draws': 1.02       # Higher odds on draws
            }
        },
        'Ladbrokes': {
            'margin': 1.08,  # 8% margin
            'variance': 0.02,  # Consistent odds
            'bias': {
                'favorites': 0.97,  # Lower odds on favorites
                'underdogs': 1.03,  # Higher odds on underdogs
                'draws': 1.01       # Slightly higher odds on draws
            }
        },
        'William Hill': {
            'margin': 1.07,  # 7% margin
            'variance': 0.03,  # Moderate variance
            'bias': {
                'favorites': 0.99,  # Slightly lower odds on favorites
                'underdogs': 1.01,  # Slightly higher odds on underdogs
                'draws': 0.99       # Slightly lower odds on draws
            }
        },
        'Paddy Power': {
            'margin': 1.09,  # 9% margin
            'variance': 0.05,  # Higher variance (more special offers)
            'bias': {
                'favorites': 0.98,  # Lower odds on favorites
                'underdogs': 1.03,  # Higher odds on underdogs
                'draws': 0.98       # Lower odds on draws
            }
        }
    }
    
    # Determine favorite/underdog
    is_team1_favorite = match_odds['team1_odds'] < match_odds['team2_odds']
    
    for source in sources:
        profile = bookmaker_profiles[source]
        
        # Apply source-specific variance
        variance = profile['variance']
        
        # Apply bookmaker bias
        if is_team1_favorite:
            team1_odds = match_odds['team1_odds'] * profile['bias']['favorites'] * np.random.uniform(1-variance, 1+variance)
            team2_odds = match_odds['team2_odds'] * profile['bias']['underdogs'] * np.random.uniform(1-variance, 1+variance)
        else:
            team1_odds = match_odds['team1_odds'] * profile['bias']['underdogs'] * np.random.uniform(1-variance, 1+variance)
            team2_odds = match_odds['team2_odds'] * profile['bias']['favorites'] * np.random.uniform(1-variance, 1+variance)
        
        draw_odds = match_odds['draw_odds'] * profile['bias']['draws'] * np.random.uniform(1-variance, 1+variance)
        
        # Apply bookmaker margin
        margin = profile['margin']
        team1_odds = team1_odds * margin
        team2_odds = team2_odds * margin
        draw_odds = draw_odds * margin
        
        # Round to 2 decimal places
        team1_odds = round(team1_odds, 2)
        team2_odds = round(team2_odds, 2)
        draw_odds = round(draw_odds, 2)
        
        # Convert implied probabilities
        team1_prob = 1 / team1_odds
        team2_prob = 1 / team2_odds
        draw_prob = 1 / draw_odds
        
        # Calculate bookmaker margin
        total_prob = team1_prob + team2_prob + draw_prob
        actual_margin = (total_prob - 1) * 100
        
        comparison[source] = {
            'team1_odds': team1_odds,
            'team2_odds': team2_odds,
            'draw_odds': draw_odds,
            'margin': f"{actual_margin:.1f}%"
        }
    
    return comparison
