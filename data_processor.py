import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

class DataProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor class for cricket data processing
        """
        self.raw_data = None
        self.processed_data = None
        self.training_data = None
        self.team_stats = {}
        self.player_stats = {}
        self.historical_matches = []
        
    def load_data(self):
        """
        Load cricket data from various sources including match history, 
        team statistics, and player performance
        
        Returns:
            bool: True if data loading is successful, False otherwise
        """
        try:
            # For this implementation, we'll create synthetic but realistic data
            # In a real implementation, this would fetch from APIs like CricAPI, ESPN Cricinfo, etc.
            
            # Create historical match data
            self._load_historical_matches()
            
            # Create team statistics
            self._generate_team_stats()
            
            # Create player statistics
            self._generate_player_stats()
            
            # Process the data for model training
            self._process_data()
            
            # Prepare training dataset
            self._prepare_training_data()
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def _load_historical_matches(self):
        """
        Load historical match data
        In a real implementation, this would fetch from cricket APIs
        """
        # Major cricket teams
        teams = [
            "India", "Australia", "England", "New Zealand", 
            "Pakistan", "South Africa", "West Indies", "Sri Lanka",
            "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland"
        ]
        
        # Generate realistic historical matches for the past 3 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 3 years of data
        
        current_date = start_date
        match_id = 1
        
        while current_date < end_date:
            # Not every day has matches
            if np.random.random() < 0.2:  # 20% chance of a match on any given day
                # Randomly select two different teams
                match_teams = np.random.choice(teams, 2, replace=False)
                team1, team2 = match_teams
                
                # Generate match result
                result_options = [team1, team2, "No Result"]
                result_probs = [0.45, 0.45, 0.1]  # 10% chance of no result/draw
                match_result = np.random.choice(result_options, p=result_probs)
                
                # Generate match details
                venues = ["Melbourne Cricket Ground", "Lord's", "Eden Gardens", "Wanderers Stadium", 
                          "Sydney Cricket Ground", "Sharjah Cricket Stadium", "Kensington Oval",
                          "R. Premadasa Stadium", "Dubai International Cricket Stadium"]
                
                match = {
                    "match_id": match_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "team1": team1,
                    "team2": team2,
                    "venue": np.random.choice(venues),
                    "result": match_result,
                    "match_type": np.random.choice(["ODI", "T20I", "Test"], p=[0.4, 0.4, 0.2]),
                    "team1_score": np.random.randint(150, 350) if match_result != "No Result" else None,
                    "team2_score": np.random.randint(150, 350) if match_result != "No Result" and match_result != team1 else None,
                    "weather": np.random.choice(["Clear", "Cloudy", "Light rain", "Heavy rain"], p=[0.7, 0.2, 0.08, 0.02]),
                    "toss_winner": np.random.choice([team1, team2]),
                    "toss_decision": np.random.choice(["bat", "field"])
                }
                
                self.historical_matches.append(match)
                match_id += 1
            
            # Move to next day
            current_date += timedelta(days=1)
        
        self.raw_data = pd.DataFrame(self.historical_matches)
        
    def _generate_team_stats(self):
        """
        Generate team statistics based on historical matches
        """
        if self.raw_data is None:
            return
        
        # Get unique teams
        all_teams = list(set(self.raw_data['team1'].unique()) | set(self.raw_data['team2'].unique()))
        
        # Initialize team stats
        for team in all_teams:
            self.team_stats[team] = {
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "no_results": 0,
                "win_percentage": 0,
                "home_wins": 0,
                "away_wins": 0,
                "form": [],  # Recent form (last 10 matches)
                "head_to_head": {}
            }
        
        # Calculate team stats from match history
        for _, match in self.raw_data.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            result = match['result']
            
            # Update total matches
            self.team_stats[team1]['total_matches'] += 1
            self.team_stats[team2]['total_matches'] += 1
            
            # Update wins, losses, no results
            if result == team1:
                self.team_stats[team1]['wins'] += 1
                self.team_stats[team1]['form'].append('W')
                self.team_stats[team2]['losses'] += 1
                self.team_stats[team2]['form'].append('L')
                
                # Update home/away wins
                if match['venue'] in ["Melbourne Cricket Ground", "Sydney Cricket Ground"] and team1 == "Australia":
                    self.team_stats[team1]['home_wins'] += 1
                elif match['venue'] in ["Lord's"] and team1 == "England":
                    self.team_stats[team1]['home_wins'] += 1
                elif match['venue'] in ["Eden Gardens"] and team1 == "India":
                    self.team_stats[team1]['home_wins'] += 1
                else:
                    self.team_stats[team1]['away_wins'] += 1
                
            elif result == team2:
                self.team_stats[team2]['wins'] += 1
                self.team_stats[team2]['form'].append('W')
                self.team_stats[team1]['losses'] += 1
                self.team_stats[team1]['form'].append('L')
                
                # Update home/away wins
                if match['venue'] in ["Melbourne Cricket Ground", "Sydney Cricket Ground"] and team2 == "Australia":
                    self.team_stats[team2]['home_wins'] += 1
                elif match['venue'] in ["Lord's"] and team2 == "England":
                    self.team_stats[team2]['home_wins'] += 1
                elif match['venue'] in ["Eden Gardens"] and team2 == "India":
                    self.team_stats[team2]['home_wins'] += 1
                else:
                    self.team_stats[team2]['away_wins'] += 1
                
            else:  # No result
                self.team_stats[team1]['no_results'] += 1
                self.team_stats[team1]['form'].append('N')
                self.team_stats[team2]['no_results'] += 1
                self.team_stats[team2]['form'].append('N')
            
            # Update head-to-head records
            if team2 not in self.team_stats[team1]['head_to_head']:
                self.team_stats[team1]['head_to_head'][team2] = {'wins': 0, 'losses': 0, 'no_results': 0}
            
            if team1 not in self.team_stats[team2]['head_to_head']:
                self.team_stats[team2]['head_to_head'][team1] = {'wins': 0, 'losses': 0, 'no_results': 0}
            
            if result == team1:
                self.team_stats[team1]['head_to_head'][team2]['wins'] += 1
                self.team_stats[team2]['head_to_head'][team1]['losses'] += 1
            elif result == team2:
                self.team_stats[team2]['head_to_head'][team1]['wins'] += 1
                self.team_stats[team1]['head_to_head'][team2]['losses'] += 1
            else:
                self.team_stats[team1]['head_to_head'][team2]['no_results'] += 1
                self.team_stats[team2]['head_to_head'][team1]['no_results'] += 1
        
        # Calculate win percentage and keep only last 10 results for form
        for team in all_teams:
            total_results = self.team_stats[team]['wins'] + self.team_stats[team]['losses']
            if total_results > 0:
                self.team_stats[team]['win_percentage'] = (self.team_stats[team]['wins'] / total_results) * 100
            
            # Keep only last 10 matches for form
            self.team_stats[team]['form'] = self.team_stats[team]['form'][-10:] if self.team_stats[team]['form'] else []
    
    def _generate_player_stats(self):
        """
        Generate player statistics
        In a real implementation, this would be based on player performance data
        """
        # Create a list of fictional but realistic cricket players
        players = {
            "India": ["Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "Ravindra Jadeja", "KL Rahul"],
            "Australia": ["Steve Smith", "Pat Cummins", "Mitchell Starc", "David Warner", "Marnus Labuschagne"],
            "England": ["Joe Root", "Ben Stokes", "Jofra Archer", "Jos Buttler", "James Anderson"],
            "New Zealand": ["Kane Williamson", "Trent Boult", "Ross Taylor", "Tim Southee", "Devon Conway"],
            "Pakistan": ["Babar Azam", "Shaheen Afridi", "Mohammad Rizwan", "Shadab Khan", "Fakhar Zaman"],
            "South Africa": ["Quinton de Kock", "Kagiso Rabada", "Anrich Nortje", "Aiden Markram", "David Miller"],
            "West Indies": ["Kieron Pollard", "Jason Holder", "Shimron Hetmyer", "Nicholas Pooran", "Shai Hope"],
            "Sri Lanka": ["Dimuth Karunaratne", "Wanindu Hasaranga", "Dushmantha Chameera", "Kusal Mendis", "Pathum Nissanka"],
            "Bangladesh": ["Shakib Al Hasan", "Mushfiqur Rahim", "Mustafizur Rahman", "Tamim Iqbal", "Mahmudullah"],
            "Afghanistan": ["Rashid Khan", "Mohammad Nabi", "Mujeeb Ur Rahman", "Rahmanullah Gurbaz", "Hazratullah Zazai"],
            "Zimbabwe": ["Sikandar Raza", "Sean Williams", "Blessing Muzarabani", "Craig Ervine", "Tendai Chatara"],
            "Ireland": ["Paul Stirling", "Andy Balbirnie", "Josh Little", "Curtis Campher", "Mark Adair"]
        }
        
        # Generate player statistics
        for team, player_list in players.items():
            for player in player_list:
                # Create player profile
                roles = ["Batsman", "Bowler", "All-rounder", "Wicket-keeper Batsman"]
                role_weights = [0.4, 0.3, 0.2, 0.1]  # More batsmen than others
                
                role = np.random.choice(roles, p=role_weights)
                
                # Generate statistics based on role
                if role in ["Batsman", "Wicket-keeper Batsman", "All-rounder"]:
                    batting_avg = np.random.normal(35, 10)  # Mean 35, std 10
                    batting_avg = max(15, min(60, batting_avg))  # Clip between 15 and 60
                    
                    strike_rate = np.random.normal(85, 15)
                    strike_rate = max(60, min(120, strike_rate))
                else:
                    batting_avg = np.random.normal(15, 5)
                    batting_avg = max(5, min(25, batting_avg))
                    
                    strike_rate = np.random.normal(70, 10)
                    strike_rate = max(50, min(90, strike_rate))
                
                if role in ["Bowler", "All-rounder"]:
                    bowling_avg = np.random.normal(25, 5)
                    bowling_avg = max(15, min(35, bowling_avg))
                    
                    economy_rate = np.random.normal(4.5, 1)
                    economy_rate = max(3, min(7, economy_rate))
                else:
                    bowling_avg = np.random.normal(40, 10)
                    bowling_avg = max(30, min(60, bowling_avg))
                    
                    economy_rate = np.random.normal(5.5, 1)
                    economy_rate = max(4, min(8, economy_rate))
                
                # Form calculations (recent performance trend)
                form_factor = np.random.normal(0, 0.2)  # Mean 0, std 0.2
                recent_batting_avg = batting_avg * (1 + form_factor)
                recent_bowling_avg = bowling_avg * (1 - form_factor if form_factor > 0 else 1 + abs(form_factor))
                
                # Final player stats
                self.player_stats[player] = {
                    "name": player,
                    "team": team,
                    "role": role,
                    "batting_avg": round(batting_avg, 2),
                    "strike_rate": round(strike_rate, 2),
                    "bowling_avg": round(bowling_avg, 2) if role in ["Bowler", "All-rounder"] else None,
                    "economy_rate": round(economy_rate, 2) if role in ["Bowler", "All-rounder"] else None,
                    "recent_form": "Good" if form_factor > 0.1 else "Average" if abs(form_factor) <= 0.1 else "Poor",
                    "recent_batting_avg": round(recent_batting_avg, 2),
                    "recent_bowling_avg": round(recent_bowling_avg, 2) if role in ["Bowler", "All-rounder"] else None,
                    "fitness": np.random.choice(["Fit", "Minor Injury", "Recovering", "Doubtful"], p=[0.85, 0.05, 0.05, 0.05])
                }
    
    def _process_data(self):
        """
        Process raw data into a format suitable for analysis and model training
        """
        if self.raw_data is None:
            return
        
        # Add team statistics to match data
        processed_data = []
        
        for _, match in self.raw_data.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            
            # Skip if team stats not available
            if team1 not in self.team_stats or team2 not in self.team_stats:
                continue
            
            # Extract head-to-head record
            h2h = self.team_stats[team1]['head_to_head'].get(team2, {'wins': 0, 'losses': 0, 'no_results': 0})
            
            # Calculate form as win percentage in last few matches
            team1_form = self.team_stats[team1]['form']
            team1_recent_win_pct = (team1_form.count('W') / len(team1_form)) * 100 if team1_form else 50
            
            team2_form = self.team_stats[team2]['form']
            team2_recent_win_pct = (team2_form.count('W') / len(team2_form)) * 100 if team2_form else 50
            
            # Check if venue is home for either team
            venues_by_country = {
                "Australia": ["Melbourne Cricket Ground", "Sydney Cricket Ground"],
                "England": ["Lord's", "Old Trafford", "Edgbaston"],
                "India": ["Eden Gardens", "Wankhede Stadium", "M. Chinnaswamy Stadium"],
                "South Africa": ["Wanderers Stadium", "SuperSport Park"],
                "New Zealand": ["Eden Park", "Basin Reserve"],
                "West Indies": ["Kensington Oval", "Sabina Park"],
                "Sri Lanka": ["R. Premadasa Stadium", "Galle International Stadium"],
                "Pakistan": ["National Stadium", "Gaddafi Stadium"],
                "Bangladesh": ["Shere Bangla National Stadium", "Zahur Ahmed Chowdhury Stadium"],
                "UAE": ["Dubai International Cricket Stadium", "Sharjah Cricket Stadium"]
            }
            
            venue = match['venue']
            team1_playing_home = any(venue in venues for country, venues in venues_by_country.items() if country in team1)
            team2_playing_home = any(venue in venues for country, venues in venues_by_country.items() if country in team2)
            
            # Process match data with features
            processed_match = {
                "match_id": match['match_id'],
                "date": match['date'],
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "result": match['result'],
                "match_type": match['match_type'],
                "team1_win_pct": self.team_stats[team1]['win_percentage'],
                "team2_win_pct": self.team_stats[team2]['win_percentage'],
                "team1_recent_form": team1_recent_win_pct,
                "team2_recent_form": team2_recent_win_pct,
                "h2h_team1_wins": h2h['wins'],
                "h2h_team2_wins": h2h['losses'],
                "h2h_no_results": h2h['no_results'],
                "team1_home_advantage": 1 if team1_playing_home else 0,
                "team2_home_advantage": 1 if team2_playing_home else 0,
                "toss_winner": match['toss_winner'],
                "toss_decision": match['toss_decision'],
                "weather": match['weather'],
                # Add more features as needed
            }
            
            processed_data.append(processed_match)
        
        self.processed_data = pd.DataFrame(processed_data)
    
    def _prepare_training_data(self):
        """
        Prepare data for model training
        """
        if self.processed_data is None:
            return
        
        # Create training dataset with features and target
        training_data = self.processed_data.copy()
        
        # Create target variable as a categorical (0 for team2 wins, 1 for team1 wins, 2 for no result/draw)
        training_data['target'] = training_data.apply(
            lambda row: 1 if row['result'] == row['team1'] else 0 if row['result'] == row['team2'] else 2,
            axis=1
        )
        
        # Create features for weather conditions
        weather_mapping = {
            'Clear': 0,
            'Cloudy': 1,
            'Light rain': 2,
            'Heavy rain': 3
        }
        
        training_data['weather_encoded'] = training_data['weather'].map(weather_mapping)
        
        # Create features for toss advantage
        training_data['team1_won_toss'] = (training_data['toss_winner'] == training_data['team1']).astype(int)
        
        # Keep only necessary columns for training
        features = [
            'team1_win_pct', 'team2_win_pct', 
            'team1_recent_form', 'team2_recent_form',
            'h2h_team1_wins', 'h2h_team2_wins', 'h2h_no_results',
            'team1_home_advantage', 'team2_home_advantage',
            'team1_won_toss', 'weather_encoded',
            'target'  # Target variable
        ]
        
        self.training_data = training_data[features]
    
    def get_processed_data(self):
        """
        Get processed cricket data
        
        Returns:
            pandas.DataFrame: Processed cricket data
        """
        return self.processed_data
    
    def get_training_data(self):
        """
        Get data prepared for model training
        
        Returns:
            pandas.DataFrame: Training data with features and target
        """
        return self.training_data
    
    def get_team_stats(self, team_name):
        """
        Get statistics for a specific team
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            dict: Team statistics
        """
        return self.team_stats.get(team_name, {})
    
    def get_player_stats(self, player_name):
        """
        Get statistics for a specific player
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            dict: Player statistics
        """
        return self.player_stats.get(player_name, {})
    
    def get_head_to_head(self, team1, team2):
        """
        Get head-to-head record between two teams
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            dict: Head-to-head statistics
        """
        if team1 in self.team_stats and team2 in self.team_stats[team1]['head_to_head']:
            return self.team_stats[team1]['head_to_head'][team2]
        return {'wins': 0, 'losses': 0, 'no_results': 0}
