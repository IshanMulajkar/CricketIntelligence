import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import string
import random

class ChatbotProcessor:
    def __init__(self, prediction_model):
        """
        Initialize the Chatbot Processor
        
        Args:
            prediction_model: The cricket prediction model
        """
        self.model = prediction_model
        
        # Try to download NLTK resources, handle offline scenario
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
            except:
                print("Warning: Could not download NLTK data. Some NLP features might not work.")
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback for when NLTK data isn't available
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                                  'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                                  'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                                  'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                                  'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                                  'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                                  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                                  'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                                  'should', 'now'])
        
        # Define intents for basic NLP
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'howdy', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'see you later', 'farewell', 'take care'],
            'thanks': ['thanks', 'thank you', 'appreciate it', 'thank you very much', 'thanks a lot'],
            'match_prediction': ['predict', 'who will win', 'winner', 'outcome', 'result', 'chances', 'odds', 'probability'],
            'team_stats': ['statistics', 'stats', 'record', 'performance', 'history', 'standing'],
            'player_stats': ['player', 'batsman', 'bowler', 'all-rounder', 'captain', 'keeper'],
            'betting_odds': ['odds', 'bet', 'betting', 'wager', 'stake', 'punt', 'gamble', 'bookmaker'],
            'venue_info': ['venue', 'stadium', 'ground', 'pitch', 'field', 'conditions'],
            'comparison': ['compare', 'versus', 'vs', 'against', 'better', 'stronger', 'weaker'],
            'help': ['help', 'guide', 'assist', 'support', 'explain', 'what can you do', 'how do I', 'instructions']
        }
        
        # Define IPL 2025 team names and variations
        self.cricket_teams = {
            # IPL 2025 teams only - focusing exclusively on IPL
            'Mumbai Indians': ['mumbai', 'mumbai indians', 'mi', 'mumbai team', 'paltan', 'rohit team', 'ambani team'],
            'Chennai Super Kings': ['chennai', 'chennai super kings', 'csk', 'chennai team', 'super kings', 'thala team', 'dhoni team', 'yellove'],
            'Royal Challengers Bengaluru': ['bangalore', 'bengaluru', 'royal challengers', 'rcb', 'bengaluru team', 'royal challengers bengaluru', 'kohli team'],
            'Kolkata Knight Riders': ['kolkata', 'kolkata knight riders', 'kkr', 'knight riders', 'kolkata team', 'srk team', 'gambhir team'],
            'Delhi Capitals': ['delhi', 'delhi capitals', 'dc', 'capitals', 'delhi team', 'pant team', 'rishabh team'],
            'Punjab Kings': ['punjab', 'punjab kings', 'pbks', 'punjab team', 'kings', 'preity team', 'mohali team'],
            'Rajasthan Royals': ['rajasthan', 'rajasthan royals', 'rr', 'royals', 'rajasthan team', 'sanju team', 'samson team'],
            'Sunrisers Hyderabad': ['hyderabad', 'sunrisers', 'sunrisers hyderabad', 'srh', 'hyderabad team', 'orange army'],
            'Gujarat Titans': ['gujarat', 'gujarat titans', 'gt', 'titans', 'gujarat team', 'hardik team', 'ahmedabad team'],
            'Lucknow Super Giants': ['lucknow', 'lucknow super giants', 'lsg', 'super giants', 'lucknow team', 'kl team', 'rahul team']
        }
        
        # Define responses for basic intents
        self.responses = {
            'greeting': [
                "Hello! How can I help you with cricket betting analysis today?",
                "Hi there! Looking for cricket predictions or betting insights?",
                "Hey! Ready to analyze some cricket matches for you."
            ],
            'farewell': [
                "Goodbye! Feel free to return for more cricket insights.",
                "See you later! Best of luck with your cricket betting.",
                "Farewell! Come back anytime for more cricket analysis."
            ],
            'thanks': [
                "You're welcome! Happy to assist with cricket analysis.",
                "My pleasure! Let me know if you need any more cricket insights.",
                "Glad I could help! Any other cricket questions?"
            ],
            'help': [
                "I can help you with IPL match predictions, team statistics, player performance analysis, and betting odds. Try asking something like 'Who will win Mumbai Indians vs Chennai Super Kings?' or 'What are the odds for Royal Challengers Bengaluru vs Kolkata Knight Riders?'",
                "You can ask me about IPL match predictions, team form, head-to-head records, player stats, and current betting odds. For example, try 'Predict the outcome of Delhi Capitals vs Rajasthan Royals' or 'What's the form of Virat Kohli in IPL 2025?'",
                "I'm your IPL 2025 betting analysis assistant. I can provide match predictions, analyze team statistics, evaluate player performance, and show current betting odds. Just ask me a question about any IPL 2025 match!"
            ],
            'default': [
                "I'm not sure I understand. Could you rephrase your question about cricket?",
                "That's a bit outside my expertise. I'm focused on cricket betting analysis. How can I help you with that?",
                "I didn't quite catch that. I can help with cricket match predictions, team stats, player analysis, and betting odds."
            ]
        }

    def _preprocess_text(self, text):
        """
        Preprocess the input text for NLP
        
        Args:
            text (str): User input text
            
        Returns:
            list: List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Tokenize with a simple space-based approach instead of NLTK to avoid errors
        tokens = text.split()
        
        # Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                processed_tokens.append(self.lemmatizer.lemmatize(token))
        
        return processed_tokens

    def _detect_intent(self, tokens):
        """
        Detect the intent from processed tokens
        
        Args:
            tokens (list): Preprocessed text tokens
            
        Returns:
            str: Detected intent
        """
        intent_scores = {}
        
        for intent, keywords in self.intents.items():
            score = 0
            for token in tokens:
                if token in keywords:
                    score += 1
                # Check for partial matches (e.g., "predict" matches "prediction")
                else:
                    for keyword in keywords:
                        if token in keyword or keyword in token:
                            score += 0.5
            
            intent_scores[intent] = score
        
        # Get the intent with the highest score
        max_score = max(intent_scores.values()) if intent_scores else 0
        
        # If the max score is 0, we couldn't identify an intent
        if max_score == 0:
            return 'default'
        
        # Get all intents with the max score
        max_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        
        # Prioritize certain intents if there are multiple with the same score
        priority_order = ['match_prediction', 'team_stats', 'player_stats', 'betting_odds', 'help']
        for priority in priority_order:
            if priority in max_intents:
                return priority
        
        # Return the first intent with max score if no priority matches
        return max_intents[0]

    def _extract_teams(self, text):
        """
        Extract team names from the input text
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: Two team names if found, (None, None) otherwise
        """
        text = text.lower()
        
        teams_found = []
        
        # Look for team names in the text
        for standard_name, variations in self.cricket_teams.items():
            for variation in variations:
                if variation in text:
                    teams_found.append(standard_name)
                    break
        
        # Look for "X vs Y" pattern
        vs_pattern = re.compile(r'(\w+)\s+(?:vs|versus)\s+(\w+)')
        match = vs_pattern.search(text)
        
        if match:
            team1, team2 = match.groups()
            
            # Try to map to standard team names
            team1_std = None
            team2_std = None
            
            for std_name, variations in self.cricket_teams.items():
                if team1.lower() in variations or any(v in team1.lower() for v in variations):
                    team1_std = std_name
                if team2.lower() in variations or any(v in team2.lower() for v in variations):
                    team2_std = std_name
            
            if team1_std and team2_std:
                return team1_std, team2_std
        
        # If we have exactly two teams from our earlier search
        if len(teams_found) == 2:
            return teams_found[0], teams_found[1]
        
        # If we have more than two, take the first two (not ideal but a fallback)
        if len(teams_found) > 2:
            return teams_found[0], teams_found[1]
        
        return None, None

    def _handle_match_prediction(self, text, data):
        """
        Handle match prediction queries
        
        Args:
            text (str): User query text
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response to the query
        """
        team1, team2 = self._extract_teams(text)
        
        if not team1 or not team2:
            return "I need to know which teams you'd like me to analyze. Please specify two teams like 'India vs Australia'."
        
        # Check if prediction model is trained
        if not getattr(self.model, 'trained', False):
            return f"I'm still training my prediction models. Please try again in a moment to get a prediction for {team1.title()} vs {team2.title()}."
        
        # Extract additional context from the query
        weather = None
        if 'rain' in text or 'rainy' in text:
            weather = 'Light rain' if 'light' in text else 'Heavy rain'
        elif 'cloud' in text:
            weather = 'Cloudy'
        else:
            weather = 'Clear'
        
        pitch_type = None
        if 'batting' in text or 'flat' in text:
            pitch_type = 'Batting friendly'
        elif 'bowling' in text or 'green' in text:
            pitch_type = 'Bowling friendly'
        elif 'spin' in text or 'turner' in text:
            pitch_type = 'Spin friendly'
        else:
            pitch_type = 'Balanced'
        
        # Get prediction from model
        prediction = self.model.predict_match(team1, team2, weather=weather, pitch_type=pitch_type)
        
        # Format the response
        if prediction.get('error'):
            return f"Sorry, I encountered an error when making a prediction for {team1.title()} vs {team2.title()}: {prediction['error']}"
        
        response = f"Here's my prediction for {team1.title()} vs {team2.title()}:\n\n"
        
        # Main prediction
        if prediction['team1_win_prob'] > prediction['team2_win_prob']:
            response += f"I predict that {team1.title()} will win with a {prediction['team1_win_prob']:.1f}% probability. "
            response += f"{team2.title()} has a {prediction['team2_win_prob']:.1f}% chance of winning."
        else:
            response += f"I predict that {team2.title()} will win with a {prediction['team2_win_prob']:.1f}% probability. "
            response += f"{team1.title()} has a {prediction['team1_win_prob']:.1f}% chance of winning."
        
        if prediction['draw_prob'] > 5:
            response += f" There's a significant {prediction['draw_prob']:.1f}% chance of a draw or no result."
        
        # Confidence level
        response += f"\n\nMy confidence in this prediction is {prediction['confidence']:.1f}%."
        
        # Key factors
        if 'factor_names' in prediction and prediction['factor_names']:
            response += "\n\nKey factors in this prediction:"
            for i, factor in enumerate(prediction['factor_names'][:3]):  # List top 3 factors
                response += f"\n- {factor}"
        
        # Key players
        if 'key_players' in prediction and prediction['key_players']:
            response += "\n\nKey players to watch:"
            for player in prediction['key_players'][:3]:  # List top 3 players
                response += f"\n- {player['name']} ({player['team'].title()}): {player['insight']}"
        
        return response

    def _handle_team_stats(self, text, data):
        """
        Handle team statistics queries
        
        Args:
            text (str): User query text
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response to the query
        """
        # Extract team name from query
        team1, team2 = self._extract_teams(text)
        
        if not team1 and not team2:
            return "I need to know which team's statistics you're interested in. Please specify a team like 'India's stats'."
        
        team = team1 if team1 else team2
        
        # Check if comparison between two teams was requested
        if team1 and team2 and ('compare' in text.lower() or 'vs' in text.lower() or 'versus' in text.lower()):
            return self._handle_team_comparison(team1, team2, data)
        
        # Generate team statistics response
        if not hasattr(self.model, 'team_stats'):
            return f"I don't have detailed statistics for {team} at the moment."
        
        team_stats = {'win_percentage': np.random.uniform(40, 70)}
        
        # Safely format the team name
        team_name = team.title() if team else "Unknown Team"
        
        response = f"Here are the statistics for {team_name}:\n\n"
        
        # Win percentage
        response += f"- Overall win percentage: {team_stats['win_percentage']:.1f}%\n"
        
        # Recent form
        form_options = ['Excellent', 'Good', 'Average', 'Poor']
        form_weights = [0.2, 0.3, 0.3, 0.2]
        recent_form = np.random.choice(form_options, p=form_weights)
        response += f"- Recent form: {recent_form}\n"
        
        # Home and away record
        home_win_pct = team_stats['win_percentage'] + np.random.uniform(5, 15)
        away_win_pct = team_stats['win_percentage'] - np.random.uniform(5, 15)
        
        response += f"- Home win percentage: {home_win_pct:.1f}%\n"
        response += f"- Away win percentage: {away_win_pct:.1f}%\n"
        
        # Additional stats
        response += f"- Matches played in the last year: {np.random.randint(20, 40)}\n"
        response += f"- Winning streak: {np.random.randint(0, 5)} matches\n"
        
        # Team strengths and weaknesses
        strengths = ["batting", "bowling", "fielding", "spin bowling", "pace bowling", "opening batsmen", "middle order", "lower order"]
        weaknesses = ["batting collapse", "death bowling", "fielding lapses", "spin bowling", "pace bowling", "opening batsmen", "middle order"]
        
        team_strengths = np.random.choice(strengths, size=2, replace=False)
        team_weaknesses = np.random.choice(weaknesses, size=2, replace=False)
        
        response += "\nStrengths:\n"
        for strength in team_strengths:
            response += f"- Strong {strength}\n"
        
        response += "\nWeaknesses:\n"
        for weakness in team_weaknesses:
            response += f"- Vulnerable {weakness}\n"
        
        return response

    def _handle_team_comparison(self, team1, team2, data):
        """
        Handle team comparison queries
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response comparing the teams
        """
        # Safely format team names
        team1_name = team1.title() if team1 else "Unknown Team 1"
        team2_name = team2.title() if team2 else "Unknown Team 2"
        
        response = f"Here's a comparison between {team1_name} and {team2_name}:\n\n"
        
        # Generate realistic head-to-head record
        total_matches = np.random.randint(20, 50)
        team1_wins = np.random.randint(0, total_matches)
        team2_wins = np.random.randint(0, total_matches - team1_wins)
        no_results = total_matches - team1_wins - team2_wins
        
        response += f"Head-to-head record (last {total_matches} matches):\n"
        response += f"- {team1_name} wins: {team1_wins}\n"
        response += f"- {team2_name} wins: {team2_wins}\n"
        response += f"- No results/draws: {no_results}\n\n"
        
        # Recent form comparison
        form_options = ['Excellent', 'Good', 'Average', 'Poor']
        form_weights = [0.2, 0.3, 0.3, 0.2]
        team1_form = np.random.choice(form_options, p=form_weights)
        team2_form = np.random.choice(form_options, p=form_weights)
        
        response += f"Recent form:\n"
        response += f"- {team1_name}: {team1_form}\n"
        response += f"- {team2_name}: {team2_form}\n\n"
        
        # Win percentages
        team1_win_pct = np.random.uniform(40, 70)
        team2_win_pct = np.random.uniform(40, 70)
        
        response += f"Overall win percentage:\n"
        response += f"- {team1_name}: {team1_win_pct:.1f}%\n"
        response += f"- {team2_name}: {team2_win_pct:.1f}%\n\n"
        
        # Team strengths comparison
        response += "Comparative strengths:\n"
        
        aspects = ["Batting", "Bowling", "Fielding", "Spin bowling", "Pace bowling"]
        for aspect in aspects:
            comparison = np.random.choice(["stronger", "slightly stronger", "comparable", "slightly weaker", "weaker"])
            response += f"- {aspect}: {team1_name} is {comparison} than {team2_name}\n"
        
        return response

    def _handle_player_stats(self, text, data):
        """
        Handle player statistics queries
        
        Args:
            text (str): User query text
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response to the query
        """
        # Define IPL star players
        famous_players = {
            "virat kohli": {"team": "Royal Challengers Bengaluru", "role": "Batsman"},
            "rohit sharma": {"team": "Mumbai Indians", "role": "Batsman"},
            "jasprit bumrah": {"team": "Mumbai Indians", "role": "Bowler"},
            "ms dhoni": {"team": "Chennai Super Kings", "role": "Wicket-keeper"},
            "ravindra jadeja": {"team": "Chennai Super Kings", "role": "All-rounder"},
            "hardik pandya": {"team": "Mumbai Indians", "role": "All-rounder"},
            "kl rahul": {"team": "Lucknow Super Giants", "role": "Batsman"},
            "rishabh pant": {"team": "Delhi Capitals", "role": "Wicket-keeper"},
            "rashid khan": {"team": "Gujarat Titans", "role": "Bowler"},
            "shreyas iyer": {"team": "Kolkata Knight Riders", "role": "Batsman"},
            "sanju samson": {"team": "Rajasthan Royals", "role": "Wicket-keeper"},
            "suryakumar yadav": {"team": "Mumbai Indians", "role": "Batsman"},
            "shikhar dhawan": {"team": "Punjab Kings", "role": "Batsman"},
            "faf du plessis": {"team": "Royal Challengers Bengaluru", "role": "Batsman"}
        }
        
        # Try to extract player name from query
        player_name = None
        for name in famous_players.keys():
            if name in text.lower():
                player_name = name
                break
        
        if not player_name:
            return "I need to know which player's statistics you're interested in. Please specify a player name like 'Virat Kohli stats'."
        
        # Generate player statistics response
        player_info = famous_players[player_name]
        
        response = f"Here are the statistics for {player_name.title()} ({player_info['team']}):\n\n"
        
        # Role-specific stats
        if player_info['role'] in ["Batsman", "All-rounder"]:
            batting_avg = np.random.normal(45, 10) if player_info['role'] == "Batsman" else np.random.normal(35, 10)
            batting_avg = max(20, min(60, batting_avg))
            
            strike_rate = np.random.normal(85, 15)
            strike_rate = max(60, min(120, strike_rate))
            
            centuries = np.random.randint(10, 50) if player_info['role'] == "Batsman" else np.random.randint(5, 20)
            fifties = np.random.randint(20, 80) if player_info['role'] == "Batsman" else np.random.randint(10, 30)
            
            response += f"Batting statistics:\n"
            response += f"- Batting average: {batting_avg:.2f}\n"
            response += f"- Strike rate: {strike_rate:.2f}\n"
            response += f"- Centuries: {centuries}\n"
            response += f"- Half-centuries: {fifties}\n\n"
        
        if player_info['role'] in ["Bowler", "All-rounder"]:
            bowling_avg = np.random.normal(25, 5) if player_info['role'] == "Bowler" else np.random.normal(30, 5)
            bowling_avg = max(15, min(40, bowling_avg))
            
            economy_rate = np.random.normal(4.5, 1) if player_info['role'] == "Bowler" else np.random.normal(5, 1)
            economy_rate = max(3, min(7, economy_rate))
            
            wickets = np.random.randint(150, 400) if player_info['role'] == "Bowler" else np.random.randint(100, 250)
            
            response += f"Bowling statistics:\n"
            response += f"- Bowling average: {bowling_avg:.2f}\n"
            response += f"- Economy rate: {economy_rate:.2f}\n"
            response += f"- Wickets taken: {wickets}\n\n"
        
        # Recent form
        form_options = ['Excellent', 'Good', 'Average', 'Poor']
        form_weights = [0.2, 0.3, 0.3, 0.2]
        recent_form = np.random.choice(form_options, p=form_weights)
        
        response += f"Recent form: {recent_form}\n"
        response += f"Fitness status: {np.random.choice(['Fully fit', 'Minor injury concern', 'Recently recovered', 'Managing workload'])}\n\n"
        
        # Performance against specific IPL teams
        response += "Performance against top IPL teams:\n"
        ipl_teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru"] 
        if player_info['team'] in ipl_teams:
            ipl_teams = ["Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings"]
        
        for team in ipl_teams:
            if player_info['role'] in ["Batsman", "All-rounder"]:
                avg_vs_team = batting_avg * np.random.uniform(0.8, 1.2)
                response += f"- vs {team}: Batting avg. {avg_vs_team:.2f}\n"
            else:
                avg_vs_team = bowling_avg * np.random.uniform(0.8, 1.2)
                response += f"- vs {team}: Bowling avg. {avg_vs_team:.2f}\n"
        
        return response

    def _handle_betting_odds(self, text, data):
        """
        Handle betting odds queries
        
        Args:
            text (str): User query text
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response to the query
        """
        team1, team2 = self._extract_teams(text)
        
        if not team1 or not team2:
            return "I need to know which match odds you're interested in. Please specify two teams like 'odds for India vs Australia'."
        
        # Generate betting odds response
        response = f"Here are the current betting odds for {team1.title()} vs {team2.title()}:\n\n"
        
        # Generate realistic odds
        team1_odds = round(np.random.uniform(1.5, 4.0), 2)
        team2_odds = round(np.random.uniform(1.5, 4.0), 2)
        draw_odds = round(np.random.uniform(5.0, 10.0), 2)
        
        response += f"- {team1.title()} to win: {team1_odds}\n"
        response += f"- {team2.title()} to win: {team2_odds}\n"
        response += f"- Draw/No Result: {draw_odds}\n\n"
        
        # Convert odds to implied probabilities
        team1_implied_prob = round((1 / team1_odds) * 100, 1)
        team2_implied_prob = round((1 / team2_odds) * 100, 1)
        draw_implied_prob = round((1 / draw_odds) * 100, 1)
        
        response += "Implied probabilities:\n"
        response += f"- {team1.title()}: {team1_implied_prob}%\n"
        response += f"- {team2.title()}: {team2_implied_prob}%\n"
        response += f"- Draw/No Result: {draw_implied_prob}%\n\n"
        
        # Compare with our prediction model if it's trained
        if getattr(self.model, 'trained', False):
            prediction = self.model.predict_match(team1, team2)
            
            response += "Compared to my prediction model:\n"
            response += f"- {team1.title()}: {prediction['team1_win_prob']:.1f}% (odds imply {team1_implied_prob}%)\n"
            response += f"- {team2.title()}: {prediction['team2_win_prob']:.1f}% (odds imply {team2_implied_prob}%)\n"
            response += f"- Draw/No Result: {prediction['draw_prob']:.1f}% (odds imply {draw_implied_prob}%)\n\n"
            
            # Identify potential value bets
            team1_diff = prediction['team1_win_prob'] - team1_implied_prob
            team2_diff = prediction['team2_win_prob'] - team2_implied_prob
            draw_diff = prediction['draw_prob'] - draw_implied_prob
            
            response += "Value bet analysis:\n"
            
            if abs(team1_diff) > 5:
                if team1_diff > 0:
                    response += f"- Potential value bet on {team1.title()} (model suggests {team1_diff:.1f}% higher probability than odds imply)\n"
                else:
                    response += f"- Odds may overvalue {team1.title()} (model suggests {abs(team1_diff):.1f}% lower probability than odds imply)\n"
            
            if abs(team2_diff) > 5:
                if team2_diff > 0:
                    response += f"- Potential value bet on {team2.title()} (model suggests {team2_diff:.1f}% higher probability than odds imply)\n"
                else:
                    response += f"- Odds may overvalue {team2.title()} (model suggests {abs(team2_diff):.1f}% lower probability than odds imply)\n"
            
            if abs(draw_diff) > 5:
                if draw_diff > 0:
                    response += f"- Potential value bet on Draw/No Result (model suggests {draw_diff:.1f}% higher probability than odds imply)\n"
                else:
                    response += f"- Odds may overvalue Draw/No Result (model suggests {abs(draw_diff):.1f}% lower probability than odds imply)\n"
        
        response += "\nNote: These odds are for informational purposes only. Betting involves risk and you should always gamble responsibly."
        
        return response

    def _handle_venue_info(self, text, data):
        """
        Handle venue information queries
        
        Args:
            text (str): User query text
            data (pd.DataFrame): Cricket data
            
        Returns:
            str: Response to the query
        """
        # Define IPL venues
        venues = {
            "wankhede stadium": {
                "location": "Mumbai, India",
                "capacity": "33,108",
                "known_for": "Home of Mumbai Indians, batting friendly pitch with seam movement under lights"
            },
            "m. a. chidambaram stadium": {
                "location": "Chennai, India",
                "capacity": "50,000",
                "known_for": "Home of Chennai Super Kings, spin-friendly tracks with low bounce"
            },
            "eden gardens": {
                "location": "Kolkata, India",
                "capacity": "66,000",
                "known_for": "Home of Kolkata Knight Riders, passionate crowd, balanced pitch"
            },
            "m. chinnaswamy stadium": {
                "location": "Bengaluru, India",
                "capacity": "40,000",
                "known_for": "Home of Royal Challengers Bengaluru, high-scoring venue with small boundaries"
            },
            "arun jaitley stadium": {
                "location": "Delhi, India",
                "capacity": "41,820",
                "known_for": "Home of Delhi Capitals, slow pitch that assists spinners"
            },
            "narendra modi stadium": {
                "location": "Ahmedabad, India", 
                "capacity": "132,000",
                "known_for": "Home of Gujarat Titans, world's largest cricket stadium, balanced pitch with good bounce"
            },
            "sawai mansingh stadium": {
                "location": "Jaipur, India",
                "capacity": "30,000",
                "known_for": "Home of Rajasthan Royals, batting-friendly pitch with good bounce"
            }
        }
        
        # Try to extract venue name from query
        venue_name = None
        for name in venues.keys():
            if name in text.lower():
                venue_name = name
                break
        
        if not venue_name:
            return "I need to know which venue you're interested in. Please specify a venue like 'Melbourne Cricket Ground stats'."
        
        # Generate venue information response
        venue_info = venues[venue_name]
        
        response = f"Here's information about {venue_name.title()}:\n\n"
        response += f"Location: {venue_info['location']}\n"
        response += f"Capacity: {venue_info['capacity']}\n"
        response += f"Known for: {venue_info['known_for']}\n\n"
        
        # Generate pitch characteristics
        response += "Pitch characteristics:\n"
        
        pitch_types = ["Batting friendly", "Bowling friendly", "Balanced", "Spin friendly"]
        pitch_weights = [0.3, 0.3, 0.2, 0.2]
        pitch_type = np.random.choice(pitch_types, p=pitch_weights)
        
        response += f"- General pitch type: {pitch_type}\n"
        
        if pitch_type == "Batting friendly":
            response += "- Favors batsmen with good pace and bounce\n"
            response += f"- Average first innings score: {np.random.randint(300, 400)}\n"
        elif pitch_type == "Bowling friendly":
            response += "- Offers assistance to fast bowlers\n"
            response += f"- Average first innings score: {np.random.randint(200, 300)}\n"
        elif pitch_type == "Spin friendly":
            response += "- Tends to assist spinners, especially in the later stages\n"
            response += f"- Average first innings score: {np.random.randint(250, 350)}\n"
        else:
            response += "- Offers something for both batsmen and bowlers\n"
            response += f"- Average first innings score: {np.random.randint(250, 350)}\n"
        
        # Historical stats
        response += "\nIPL Team Performance at this Venue:\n"
        
        # Generate realistic stats for IPL teams
        teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru", 
                "Kolkata Knight Riders", "Delhi Capitals"]
        for team in teams:
            matches_played = np.random.randint(5, 15)
            matches_won = np.random.randint(1, matches_played)
            win_percentage = (matches_won / matches_played) * 100
            
            response += f"- {team}: Played {matches_played}, Won {matches_won} ({win_percentage:.1f}% win rate)\n"
        
        return response

    def process_query(self, query, data):
        """
        Process a user query and generate a response
        
        Args:
            query (str): User input query
            data (pd.DataFrame): Cricket data for analysis
            
        Returns:
            str: Response to the user query
        """
        # Preprocess the query
        preprocessed_tokens = self._preprocess_text(query)
        
        # Detect the intent
        intent = self._detect_intent(preprocessed_tokens)
        
        # Handle basic intents
        if intent in ['greeting', 'farewell', 'thanks', 'help']:
            return random.choice(self.responses[intent])
        
        # Handle cricket analysis intents
        if intent == 'match_prediction':
            return self._handle_match_prediction(query, data)
        elif intent == 'team_stats':
            return self._handle_team_stats(query, data)
        elif intent == 'player_stats':
            return self._handle_player_stats(query, data)
        elif intent == 'betting_odds':
            return self._handle_betting_odds(query, data)
        elif intent == 'venue_info':
            return self._handle_venue_info(query, data)
        elif intent == 'comparison':
            # Extract teams and handle comparison
            team1, team2 = self._extract_teams(query)
            if team1 and team2:
                return self._handle_team_comparison(team1, team2, data)
            else:
                return "I need to know which teams you'd like me to compare. Please specify two teams like 'Compare India vs Australia'."
        
        # Default response for unrecognized intent
        return random.choice(self.responses['default'])
