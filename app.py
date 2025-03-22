import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from data_processor import DataProcessor
from ml_models import CricketPredictor
from chatbot import ChatbotProcessor
from visualization import create_match_prediction_chart, create_player_performance_chart
from odds_fetcher import get_current_odds
from utils import format_team_name, get_upcoming_matches, calculate_kelly_criterion

# Set page configuration
st.set_page_config(
    page_title="Cricket Betting Analysis Assistant",
    page_icon="🏏",
    layout="wide"
)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Initialize components
@st.cache_resource
def load_components():
    data_processor = DataProcessor()
    model = CricketPredictor()
    chatbot = ChatbotProcessor(model)
    return data_processor, model, chatbot

data_processor, model, chatbot = load_components()

# App header
st.title("🏏 Cricket Betting Analysis Assistant")
st.markdown("""
This application uses machine learning to analyze cricket matches and provide
betting insights. Ask questions about upcoming matches, team performance, or betting odds.
""")

# Main layout with tabs
tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Match Analysis", "Betting Odds"])

with tab1:
    st.header("Cricket Analysis Chatbot")
    
    # Load and train model on first run
    if not st.session_state.data_loaded:
        with st.spinner("Loading cricket data..."):
            success = data_processor.load_data()
            if success:
                st.session_state.data_loaded = True
                st.success("Cricket data loaded successfully!")
            else:
                st.error("Failed to load cricket data. Please try again later.")

    if st.session_state.data_loaded and not st.session_state.model_trained:
        with st.spinner("Training prediction models..."):
            success = model.train(data_processor.get_training_data())
            if success:
                st.session_state.model_trained = True
                st.success("Prediction models trained successfully!")
            else:
                st.error("Failed to train prediction models. Please try again later.")
    
    # Chat interface
    st.subheader("Ask me about cricket matches and betting insights")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
    
    # User input
    user_query = st.text_input("Type your question here (e.g., 'Who will win India vs Australia?')", key="user_input")
    
    if st.button("Send") and user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get response from chatbot
        with st.spinner("Analyzing..."):
            response = chatbot.process_query(user_query, data_processor.get_processed_data())
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input and rerun to update chat
        st.rerun()

with tab2:
    st.header("Match Prediction Analysis")
    
    if st.session_state.data_loaded and st.session_state.model_trained:
        # Team selection
        upcoming_matches = get_upcoming_matches()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select teams for prediction")
            match_option = st.selectbox(
                "Choose a match",
                options=upcoming_matches,
                format_func=lambda x: f"{x['team1']} vs {x['team2']} - {x['date']}"
            )
            
            if match_option:
                team1 = match_option['team1']
                team2 = match_option['team2']
                venue = match_option.get('venue', 'Neutral')
                city = match_option.get('city', '')
                match_type = match_option.get('match_type', 'ODI')
                
                # Get match-specific pitch type and weather forecast
                default_pitch_type = match_option.get('pitch_type', 'Balanced')
                default_weather = match_option.get('weather', 'Clear')
                
                # Additional match factors - select relevant defaults based on match
                suggested_factors = []
                
                # Add home advantage if venue is home for one of teams
                if venue in ["Eden Gardens", "Wankhede Stadium", "M. Chinnaswamy Stadium"] and team1 == "India":
                    suggested_factors.append("Home advantage")
                elif venue in ["Melbourne Cricket Ground", "Sydney Cricket Ground"] and team1 == "Australia":
                    suggested_factors.append("Home advantage")
                elif venue in ["Lord's", "The Oval"] and team1 == "England":
                    suggested_factors.append("Home advantage")
                # Similar checks for team2
                if venue in ["Eden Gardens", "Wankhede Stadium", "M. Chinnaswamy Stadium"] and team2 == "India":
                    suggested_factors.append("Home advantage")
                elif venue in ["Melbourne Cricket Ground", "Sydney Cricket Ground"] and team2 == "Australia":
                    suggested_factors.append("Home advantage")
                elif venue in ["Lord's", "The Oval"] and team2 == "England":
                    suggested_factors.append("Home advantage")
                
                # Add recent form by default
                suggested_factors.append("Recent form")
                
                # Add head-to-head history
                suggested_factors.append("Head-to-head history")
                
                # Remove duplicates
                suggested_factors = list(dict.fromkeys(suggested_factors))
                
                # Display multiselect with suggested factors pre-selected
                factors = st.multiselect(
                    "Select additional factors to consider",
                    ["Home advantage", "Recent form", "Head-to-head history", "Player injuries", 
                     "Pitch conditions", "Weather impact", "Tournament importance"],
                    default=suggested_factors
                )
                
                # Display match information
                st.markdown(f"""
                **Match Information:**
                - **Type:** {match_type}
                - **Venue:** {venue}, {city}
                - **Date:** {match_option['date']}
                """)
        
        with col2:
            st.subheader("Match conditions")
            
            # Weather forecast with default from match data
            weather = st.selectbox(
                "Weather forecast", 
                ["Clear", "Cloudy", "Light rain", "Heavy rain"], 
                index=["Clear", "Cloudy", "Light rain", "Heavy rain"].index(default_weather)
            )
            
            # Pitch type with default from match data
            pitch_type = st.selectbox(
                "Pitch type", 
                ["Batting friendly", "Bowling friendly", "Balanced", "Spin friendly"],
                index=["Batting friendly", "Bowling friendly", "Balanced", "Spin friendly"].index(default_pitch_type)
            )
            
            # Generate prediction
            if st.button("Generate Prediction"):
                with st.spinner("Analyzing match data and generating prediction..."):
                    prediction = model.predict_match(team1, team2, venue, weather, pitch_type, factors)
                    
                    # Display prediction results
                    st.subheader(f"Match Prediction: {team1} vs {team2}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = create_match_prediction_chart(prediction)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Key Insights")
                        st.markdown(f"- **Win probability for {team1}**: {prediction['team1_win_prob']:.1f}%")
                        st.markdown(f"- **Win probability for {team2}**: {prediction['team2_win_prob']:.1f}%")
                        st.markdown(f"- **Draw/No Result probability**: {prediction['draw_prob']:.1f}%")
                        
                        st.markdown("### Confidence Level")
                        st.progress(prediction['confidence'] / 100)
                        st.caption(f"Confidence score: {prediction['confidence']}/100")
                    
                    # Additional insights
                    st.subheader("Performance Factors")
                    factors_df = pd.DataFrame({
                        'Factor': prediction['factor_names'],
                        'Impact': prediction['factor_values']
                    })
                    
                    st.dataframe(factors_df, use_container_width=True)
                    
                    # Key player analysis
                    if 'key_players' in prediction:
                        st.subheader("Key Players to Watch")
                        for player in prediction['key_players']:
                            st.markdown(f"- **{player['name']}** ({player['team']}): {player['insight']}")
    else:
        st.info("Please wait for the data to load and models to be trained.")

with tab3:
    st.header("Current Betting Odds")
    
    if st.session_state.data_loaded:
        # Get upcoming matches for consistent data across tabs
        upcoming_matches = get_upcoming_matches()
        
        # Fetch current betting odds
        with st.spinner("Fetching latest betting odds..."):
            try:
                odds_data = get_current_odds()
                
                if odds_data:
                    # Add date to display in the table
                    for odds in odds_data:
                        # Find the match in upcoming_matches to get additional data
                        for match in upcoming_matches:
                            if match['team1'] == odds['team1'] and match['team2'] == odds['team2']:
                                odds['date'] = match['date']
                                odds['venue'] = match.get('venue', 'Neutral')
                                odds['city'] = match.get('city', '')
                                odds['match_type'] = match.get('match_type', 'ODI')
                                break
                    
                    # Display odds in a table with more information
                    st.subheader("Latest Match Odds")
                    odds_df = pd.DataFrame(odds_data)
                    
                    # Format the dataframe - include date and match type
                    formatted_df = odds_df[['match', 'date', 'match_type', 'team1', 'team1_odds', 'team2', 'team2_odds', 'draw_odds', 'source']]
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # Group by date for better organization
                    st.subheader("Upcoming Matches by Date")
                    
                    # Get unique dates
                    unique_dates = sorted(list(set(odds_df['date'].tolist())))
                    
                    # Create tabs for each date
                    date_tabs = st.tabs([f"{date}" for date in unique_dates])
                    
                    for i, date in enumerate(unique_dates):
                        with date_tabs[i]:
                            date_matches = odds_df[odds_df['date'] == date]
                            st.write(f"**{len(date_matches)} matches on {date}**")
                            
                            # Create columns for each match
                            match_cols = st.columns(min(3, len(date_matches)))
                            
                            for j, (_, match) in enumerate(date_matches.iterrows()):
                                col_idx = j % len(match_cols)
                                with match_cols[col_idx]:
                                    st.write(f"**{match['team1']} vs {match['team2']}**")
                                    st.write(f"*{match.get('match_type', 'ODI')} at {match.get('venue', 'TBD')}, {match.get('city', '')}*")
                                    st.write(f"{match['team1']}: {match['team1_odds']}")
                                    st.write(f"{match['team2']}: {match['team2_odds']}")
                                    st.write(f"Draw: {match['draw_odds']}")
                                    st.write(f"Source: {match['source']}")
                    
                    # Select a match for detailed odds analysis
                    selected_match = st.selectbox(
                        "Select a match for detailed odds analysis",
                        options=odds_df['match'].tolist(),
                        format_func=lambda x: f"{x} - {odds_df[odds_df['match'] == x]['date'].iloc[0]}"
                    )
                    
                    if selected_match:
                        match_data = odds_df[odds_df['match'] == selected_match].iloc[0]
                        
                        # Display detailed match information
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Match Details")
                            st.write(f"**Date:** {match_data['date']}")
                            st.write(f"**Venue:** {match_data.get('venue', 'TBD')}, {match_data.get('city', '')}")
                            st.write(f"**Match Type:** {match_data.get('match_type', 'ODI')}")
                            st.write(f"**Bookmaker:** {match_data['source']}")
                            st.write(f"**Last Updated:** {match_data.get('last_updated', 'N/A')}")
                        
                        with col2:
                            st.subheader("Odds")
                            st.write(f"**{match_data['team1']}:** {match_data['team1_odds']}")
                            st.write(f"**{match_data['team2']}:** {match_data['team2_odds']}")
                            st.write(f"**Draw/No Result:** {match_data['draw_odds']}")
                            
                            # Display implied probabilities
                            team1_implied_prob = (1 / match_data['team1_odds']) * 100
                            team2_implied_prob = (1 / match_data['team2_odds']) * 100
                            draw_implied_prob = (1 / match_data['draw_odds']) * 100 if match_data['draw_odds'] > 0 else 0
                            
                            # Normalize to 100%
                            total_implied = team1_implied_prob + team2_implied_prob + draw_implied_prob
                            team1_implied_prob = (team1_implied_prob / total_implied) * 100
                            team2_implied_prob = (team2_implied_prob / total_implied) * 100
                            draw_implied_prob = (draw_implied_prob / total_implied) * 100
                            
                            st.write("**Implied Probabilities:**")
                            st.write(f"- {match_data['team1']}: {team1_implied_prob:.1f}%")
                            st.write(f"- {match_data['team2']}: {team2_implied_prob:.1f}%")
                            st.write(f"- Draw/No Result: {draw_implied_prob:.1f}%")
                        
                        # Compare with our prediction if model is trained
                        if st.session_state.model_trained:
                            st.subheader("Odds Analysis")
                            
                            # Extract teams from the match
                            team1, team2 = match_data['team1'], match_data['team2']
                            
                            # Get match-specific data for better prediction
                            venue = match_data.get('venue', 'Neutral')
                            weather = None
                            pitch_type = None
                            
                            # Find additional match info from upcoming_matches
                            for match in upcoming_matches:
                                if match['team1'] == team1 and match['team2'] == team2:
                                    weather = match.get('weather', 'Clear')
                                    pitch_type = match.get('pitch_type', 'Balanced')
                                    break
                            
                            # Get our model's prediction with more details
                            prediction = model.predict_match(team1, team2, venue, weather, pitch_type)
                            
                            # Calculate value bets
                            team1_value = prediction['team1_win_prob'] - team1_implied_prob
                            team2_value = prediction['team2_win_prob'] - team2_implied_prob
                            draw_value = prediction['draw_prob'] - draw_implied_prob
                            
                            # Display comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Model vs. Bookmaker Probabilities")
                                comparison_data = {
                                    'Outcome': [f"{team1} Win", f"{team2} Win", "Draw/No Result"],
                                    'Our Model': [
                                        f"{prediction['team1_win_prob']:.1f}%", 
                                        f"{prediction['team2_win_prob']:.1f}%", 
                                        f"{prediction['draw_prob']:.1f}%"
                                    ],
                                    'Bookmaker Implied': [
                                        f"{team1_implied_prob:.1f}%", 
                                        f"{team2_implied_prob:.1f}%", 
                                        f"{draw_implied_prob:.1f}%"
                                    ],
                                    'Value': [
                                        f"{team1_value:+.1f}%", 
                                        f"{team2_value:+.1f}%", 
                                        f"{draw_value:+.1f}%"
                                    ]
                                }
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Add a plot comparing probabilities
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    x=['Model', 'Bookmaker'],
                                    y=[prediction['team1_win_prob'], team1_implied_prob],
                                    name=f"{team1} Win",
                                    marker_color='#1f77b4'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    x=['Model', 'Bookmaker'],
                                    y=[prediction['team2_win_prob'], team2_implied_prob],
                                    name=f"{team2} Win",
                                    marker_color='#ff7f0e'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    x=['Model', 'Bookmaker'],
                                    y=[prediction['draw_prob'], draw_implied_prob],
                                    name=f"Draw/No Result",
                                    marker_color='#2ca02c'
                                ))
                                
                                fig.update_layout(
                                    title="Probability Comparison",
                                    xaxis_title="Source",
                                    yaxis_title="Probability (%)",
                                    barmode='group',
                                    yaxis=dict(range=[0, 100])
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("### Value Bet Analysis")
                                
                                # Determine if there's value
                                value_threshold = 5.0  # 5% difference is considered significant value
                                
                                if abs(team1_value) > value_threshold:
                                    if team1_value > 0:
                                        st.success(f"✅ Potential value bet on {team1} to win")
                                        st.markdown(f"Our model gives {team1} a {prediction['team1_win_prob']:.1f}% chance to win, "
                                                   f"while the odds imply only {team1_implied_prob:.1f}%")
                                        
                                        # Show detailed Kelly criterion calculation
                                        prob = prediction['team1_win_prob'] / 100
                                        kelly_pct = calculate_kelly_criterion(prob, match_data['team1_odds'], 0.5)
                                        st.markdown(f"**Kelly Criterion Recommendation:** Bet {kelly_pct:.1f}% of bankroll")
                                    else:
                                        st.warning(f"⚠️ Avoid betting on {team1} to win")
                                        st.markdown(f"The odds overvalue {team1}'s chances ({team1_implied_prob:.1f}% implied vs. "
                                                   f"{prediction['team1_win_prob']:.1f}% by our model)")
                                
                                if abs(team2_value) > value_threshold:
                                    if team2_value > 0:
                                        st.success(f"✅ Potential value bet on {team2} to win")
                                        st.markdown(f"Our model gives {team2} a {prediction['team2_win_prob']:.1f}% chance to win, "
                                                   f"while the odds imply only {team2_implied_prob:.1f}%")
                                        
                                        # Show detailed Kelly criterion calculation
                                        prob = prediction['team2_win_prob'] / 100
                                        kelly_pct = calculate_kelly_criterion(prob, match_data['team2_odds'], 0.5)
                                        st.markdown(f"**Kelly Criterion Recommendation:** Bet {kelly_pct:.1f}% of bankroll")
                                    else:
                                        st.warning(f"⚠️ Avoid betting on {team2} to win")
                                        st.markdown(f"The odds overvalue {team2}'s chances ({team2_implied_prob:.1f}% implied vs. "
                                                   f"{prediction['team2_win_prob']:.1f}% by our model)")
                                
                                if abs(draw_value) > value_threshold:
                                    if draw_value > 0:
                                        st.success(f"✅ Potential value bet on Draw/No Result")
                                        st.markdown(f"Our model gives Draw/No Result a {prediction['draw_prob']:.1f}% chance, "
                                                   f"while the odds imply only {draw_implied_prob:.1f}%")
                                        
                                        # Show detailed Kelly criterion calculation
                                        prob = prediction['draw_prob'] / 100
                                        kelly_pct = calculate_kelly_criterion(prob, match_data['draw_odds'], 0.5)
                                        st.markdown(f"**Kelly Criterion Recommendation:** Bet {kelly_pct:.1f}% of bankroll")
                                    else:
                                        st.warning(f"⚠️ Avoid betting on Draw/No Result")
                                        st.markdown(f"The odds overvalue Draw/No Result chances ({draw_implied_prob:.1f}% implied vs. "
                                                   f"{prediction['draw_prob']:.1f}% by our model)")
                                
                                if (abs(team1_value) <= value_threshold and 
                                    abs(team2_value) <= value_threshold and 
                                    abs(draw_value) <= value_threshold):
                                    st.info("No significant value bets found for this match. The bookmaker odds align with our model predictions.")
                                    
                                # Add bookmaker margin information
                                bookmaker_margin = (1/match_data['team1_odds'] + 1/match_data['team2_odds'] + 1/match_data['draw_odds'] - 1) * 100
                                st.markdown(f"**Bookmaker Margin:** {bookmaker_margin:.1f}%")
                                st.caption("Lower margin means better value for bettors")
                else:
                    st.error("No odds data available at the moment. Please try again later.")
            except Exception as e:
                st.error(f"Error fetching betting odds: {str(e)}")
    else:
        st.info("Please wait for the data to load.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
Cricket Betting Analysis Assistant | For informational purposes only | Not financial advice
</div>
""", unsafe_allow_html=True)
