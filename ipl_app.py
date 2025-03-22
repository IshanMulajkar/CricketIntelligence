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
    page_title="IPL 2025 Betting Analysis Assistant",
    page_icon="🏏",
    layout="wide"
)

# Set up some custom styling for IPL theme
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .st-emotion-cache-1v0mbdj {
        background-color: #1e293b;
    }
    .st-emotion-cache-1wrcr25 {
        background-color: #334155;
    }
    h1, h2, h3 {
        color: #f97316 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #334155;
        border-radius: 4px 4px 0 0;
        padding: 0px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f97316 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

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
st.title("🏏 IPL 2025 Betting Analysis Assistant")
st.markdown("""
This application uses machine learning to analyze IPL 2025 matches and provide
betting insights. Ask questions about upcoming matches, team performance, or betting odds.
""")

# Main layout with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat Assistant", "Match Analysis", "Betting Odds", "Team Statistics"])

with tab1:
    st.header("IPL Betting Chatbot")
    
    # Load and train model on first run
    if not st.session_state.data_loaded:
        with st.spinner("Loading IPL data..."):
            success = data_processor.load_data()
            if success:
                st.session_state.data_loaded = True
                st.success("IPL data loaded successfully!")
            else:
                st.error("Failed to load IPL data. Please try again later.")

    if st.session_state.data_loaded and not st.session_state.model_trained:
        with st.spinner("Training prediction models..."):
            success = model.train(data_processor.get_training_data())
            if success:
                st.session_state.model_trained = True
                st.success("Prediction models trained successfully!")
            else:
                st.error("Failed to train prediction models. Please try again later.")
    
    # Chat interface
    st.subheader("Ask me about IPL 2025 matches and betting insights")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #0f172a; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <p style="color: white; font-weight: bold;">You:</p>
                    <p style="color: white;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #334155; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <p style="color: #f97316; font-weight: bold;">IPL Assistant:</p>
                    <p style="color: white;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # User input
    user_query = st.text_input("Type your question here (e.g., 'Who will win Mumbai Indians vs Chennai Super Kings?')", key="user_input")
    
    if st.button("Send", type="primary") and user_query:
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
    st.header("IPL Match Prediction Analysis")
    
    if st.session_state.data_loaded and st.session_state.model_trained:
        # Team selection
        upcoming_matches = get_upcoming_matches()  # Already returns IPL matches
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select IPL match for prediction")
            
            # Add filtering options
            date_filter = st.radio("Filter matches by:", ["All upcoming", "Today", "Tomorrow", "This week"])
            
            filtered_matches = upcoming_matches
            today = pd.Timestamp.now().date()
            
            if date_filter == "Today":
                filtered_matches = [m for m in upcoming_matches if pd.Timestamp(m['date']).date() == today]
            elif date_filter == "Tomorrow":
                tomorrow = today + pd.Timedelta(days=1)
                filtered_matches = [m for m in upcoming_matches if pd.Timestamp(m['date']).date() == tomorrow]
            elif date_filter == "This week":
                week_end = today + pd.Timedelta(days=7)
                filtered_matches = [m for m in upcoming_matches if today <= pd.Timestamp(m['date']).date() <= week_end]
            
            if not filtered_matches:
                st.warning("No matches found for the selected filter. Showing all matches.")
                filtered_matches = upcoming_matches
            
            match_option = st.selectbox(
                "Choose a match",
                options=filtered_matches,
                format_func=lambda x: f"{x['team1']} vs {x['team2']} - {x['date']} ({x.get('time', 'TBD')})"
            )
            
            if match_option:
                team1 = match_option['team1']
                team2 = match_option['team2']
                venue = match_option.get('venue', 'Neutral')
                city = match_option.get('city', '')
                match_type = match_option.get('match_type', 'T20')
                match_time = match_option.get('time', 'TBD')
                tournament = match_option.get('tournament', 'IPL 2025')
                
                # Get match-specific pitch type and weather forecast
                default_pitch_type = match_option.get('pitch_type', 'Balanced')
                default_weather = match_option.get('weather', 'Clear')
                
                # Additional match factors - select relevant defaults based on match
                suggested_factors = []
                
                # Add home advantage for IPL teams at their home grounds
                if "Mumbai" in venue and team1 == "Mumbai Indians":
                    suggested_factors.append("Home advantage")
                elif "Chennai" in venue and team1 == "Chennai Super Kings":
                    suggested_factors.append("Home advantage")
                elif "Chinnaswamy" in venue and team1 == "Royal Challengers Bengaluru":
                    suggested_factors.append("Home advantage")
                elif "Eden Gardens" in venue and team1 == "Kolkata Knight Riders":
                    suggested_factors.append("Home advantage")
                # Similar checks for team2
                if "Mumbai" in venue and team2 == "Mumbai Indians":
                    suggested_factors.append("Home advantage")
                elif "Chennai" in venue and team2 == "Chennai Super Kings":
                    suggested_factors.append("Home advantage")
                elif "Chinnaswamy" in venue and team2 == "Royal Challengers Bengaluru":
                    suggested_factors.append("Home advantage")
                elif "Eden Gardens" in venue and team2 == "Kolkata Knight Riders":
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
                     "Pitch conditions", "Weather impact", "Team composition", "Toss result"],
                    default=suggested_factors
                )
                
                # Display match information
                st.markdown(f"""
                **Match Information:**
                - **Tournament:** {tournament}
                - **Venue:** {venue}, {city}
                - **Date:** {match_option['date']}
                - **Time:** {match_time}
                - **Head-to-Head:** {match_option.get('head_to_head', 'Data not available')}
                """)
                
                # Show team forms
                team1_form = match_option.get('team1_form', 'Average')
                team2_form = match_option.get('team2_form', 'Average')
                
                col1a, col1b = st.columns(2)
                with col1a:
                    st.markdown(f"**{team1} Form:** {team1_form}")
                with col1b:
                    st.markdown(f"**{team2} Form:** {team2_form}")
        
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
            
            # Toss result
            toss_result = st.radio(
                "Toss winner",
                [team1, team2, "Unknown"]
            )
            
            if toss_result != "Unknown":
                toss_decision = st.radio(
                    "Toss decision",
                    ["Bat", "Bowl"]
                )
                factors.append(f"Toss: {toss_result} chose to {toss_decision}")
            
            # Generate prediction
            if st.button("Generate Prediction", type="primary"):
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
    st.header("IPL 2025 Betting Odds")
    
    if st.session_state.data_loaded:
        # Get upcoming matches for consistent data across tabs
        upcoming_matches = get_upcoming_matches()
        
        # Fetch current betting odds
        with st.spinner("Fetching latest IPL betting odds..."):
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
                                odds['match_type'] = match.get('match_type', 'T20')
                                odds['time'] = match.get('time', 'TBD')
                                break
                    
                    # Display odds in a table with more information
                    st.subheader("Latest IPL Match Odds")
                    odds_df = pd.DataFrame(odds_data)
                    
                    # Format the dataframe - include date and match type
                    formatted_df = odds_df[['match', 'date', 'time', 'team1', 'team1_odds', 'team2', 'team2_odds', 'draw_odds', 'source']]
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # Group by date for better organization
                    st.subheader("Upcoming IPL Matches by Date")
                    
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
                                    st.markdown(f"""
                                    <div style="background-color: #334155; padding: 10px; border-radius: 10px;">
                                        <h4 style="color: #f97316; text-align: center;">{match['team1']} vs {match['team2']}</h4>
                                        <p style="color: white; text-align: center;">
                                            <small>{match.get('time', 'TBD')} • {match.get('venue', 'TBD')}</small>
                                        </p>
                                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                                            <div style="text-align: center; flex: 1;">
                                                <p style="color: white; margin: 0;">{match['team1']}</p>
                                                <p style="color: #f97316; font-weight: bold; margin: 0;">{match['team1_odds']}</p>
                                            </div>
                                            <div style="text-align: center; flex: 1;">
                                                <p style="color: white; margin: 0;">Draw</p>
                                                <p style="color: #f97316; font-weight: bold; margin: 0;">{match['draw_odds']}</p>
                                            </div>
                                            <div style="text-align: center; flex: 1;">
                                                <p style="color: white; margin: 0;">{match['team2']}</p>
                                                <p style="color: #f97316; font-weight: bold; margin: 0;">{match['team2_odds']}</p>
                                            </div>
                                        </div>
                                        <p style="color: white; text-align: center; margin-top: 10px;">
                                            <small>Source: {match['source']}</small>
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Select a match for detailed odds analysis
                    selected_match = st.selectbox(
                        "Select a match for detailed odds analysis",
                        options=odds_df['match'].tolist(),
                        format_func=lambda x: f"{x} - {odds_df[odds_df['match'] == x]['date'].iloc[0]} ({odds_df[odds_df['match'] == x]['time'].iloc[0] if 'time' in odds_df.columns else 'TBD'})"
                    )
                    
                    if selected_match:
                        match_data = odds_df[odds_df['match'] == selected_match].iloc[0]
                        
                        # Display detailed match information
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Match Details")
                            
                            # Create a more visually appealing match details card
                            st.markdown(f"""
                            <div style="background-color: #334155; padding: 15px; border-radius: 10px;">
                                <h3 style="color: #f97316; text-align: center; margin-bottom: 15px;">{match_data['team1']} vs {match_data['team2']}</h3>
                                <p style="color: white;"><strong>Date:</strong> {match_data['date']}</p>
                                <p style="color: white;"><strong>Time:</strong> {match_data.get('time', 'TBD')}</p>
                                <p style="color: white;"><strong>Venue:</strong> {match_data.get('venue', 'TBD')}, {match_data.get('city', '')}</p>
                                <p style="color: white;"><strong>Match Type:</strong> {match_data.get('match_type', 'T20')}</p>
                                <p style="color: white;"><strong>Bookmaker:</strong> {match_data['source']}</p>
                                <p style="color: white;"><strong>Last Updated:</strong> {match_data.get('last_updated', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("Odds")
                            
                            # Display odds in a more visually appealing format
                            team1_implied_prob = (1 / match_data['team1_odds']) * 100
                            team2_implied_prob = (1 / match_data['team2_odds']) * 100
                            draw_implied_prob = (1 / match_data['draw_odds']) * 100 if match_data['draw_odds'] > 0 else 0
                            
                            # Normalize to 100%
                            total_implied = team1_implied_prob + team2_implied_prob + draw_implied_prob
                            team1_implied_prob = (team1_implied_prob / total_implied) * 100
                            team2_implied_prob = (team2_implied_prob / total_implied) * 100
                            draw_implied_prob = (draw_implied_prob / total_implied) * 100
                            
                            # Create a card for the odds
                            col2a, col2b, col2c = st.columns(3)
                            
                            with col2a:
                                st.markdown(f"""
                                <div style="background-color: #334155; padding: 10px; border-radius: 10px; text-align: center;">
                                    <h4 style="color: white;">{match_data['team1']}</h4>
                                    <h2 style="color: #f97316;">{match_data['team1_odds']}</h2>
                                    <p style="color: white;">Implied: {team1_implied_prob:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2b:
                                st.markdown(f"""
                                <div style="background-color: #334155; padding: 10px; border-radius: 10px; text-align: center;">
                                    <h4 style="color: white;">Draw</h4>
                                    <h2 style="color: #f97316;">{match_data['draw_odds']}</h2>
                                    <p style="color: white;">Implied: {draw_implied_prob:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2c:
                                st.markdown(f"""
                                <div style="background-color: #334155; padding: 10px; border-radius: 10px; text-align: center;">
                                    <h4 style="color: white;">{match_data['team2']}</h4>
                                    <h2 style="color: #f97316;">{match_data['team2_odds']}</h2>
                                    <p style="color: white;">Implied: {team2_implied_prob:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
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
                                    marker_color='#f97316'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    x=['Model', 'Bookmaker'],
                                    y=[prediction['team2_win_prob'], team2_implied_prob],
                                    name=f"{team2} Win",
                                    marker_color='#0ea5e9'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    x=['Model', 'Bookmaker'],
                                    y=[prediction['draw_prob'], draw_implied_prob],
                                    name=f"Draw/No Result",
                                    marker_color='#a3e635'
                                ))
                                
                                fig.update_layout(
                                    title="Probability Comparison",
                                    xaxis_title="Source",
                                    yaxis_title="Probability (%)",
                                    barmode='group',
                                    yaxis=dict(range=[0, 100]),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white')
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

with tab4:
    st.header("IPL 2025 Team Statistics")
    
    if st.session_state.data_loaded:
        # Get IPL teams
        ipl_teams = [
            "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru", 
            "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings", 
            "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans", 
            "Lucknow Super Giants"
        ]
        
        # Team selection
        selected_team = st.selectbox("Select a team", ipl_teams)
        
        if selected_team:
            # Display team logo and info
            st.markdown(f"### {selected_team} Statistics")
            
            # Create tabs for different statistics
            team_tabs = st.tabs(["Performance", "Players", "Venue Records", "Head-to-Head"])
            
            # Sample team performance data (would come from data_processor in a real implementation)
            team_performance = {
                "matches_played": 14,
                "wins": 9,
                "losses": 5,
                "points": 18,
                "net_run_rate": 0.45,
                "batting_avg": 168,
                "bowling_avg": 152,
                "highest_score": 223,
                "lowest_score": 131
            }
            
            # Sample player data
            players = [
                {"name": "Player 1", "role": "Batsman", "matches": 14, "runs": 420, "average": 35.0, "strike_rate": 145.0},
                {"name": "Player 2", "role": "Bowler", "matches": 14, "wickets": 18, "economy": 7.8, "average": 22.3},
                {"name": "Player 3", "role": "All-rounder", "matches": 14, "runs": 280, "wickets": 12, "batting_average": 28.0, "bowling_average": 24.5},
                {"name": "Player 4", "role": "Batsman", "matches": 12, "runs": 320, "average": 32.0, "strike_rate": 138.0},
                {"name": "Player 5", "role": "Bowler", "matches": 14, "wickets": 15, "economy": 8.2, "average": 24.1}
            ]
            
            # Sample venue records
            venue_records = [
                {"venue": "Wankhede Stadium", "matches": 8, "wins": 6, "losses": 2, "batting_avg": 175, "bowling_avg": 158},
                {"venue": "Eden Gardens", "matches": 5, "wins": 3, "losses": 2, "batting_avg": 162, "bowling_avg": 155},
                {"venue": "M. Chinnaswamy Stadium", "matches": 4, "wins": 2, "losses": 2, "batting_avg": 182, "bowling_avg": 175}
            ]
            
            # Sample head-to-head data
            h2h_data = {}
            for team in ipl_teams:
                if team != selected_team:
                    h2h_data[team] = {
                        "matches": np.random.randint(10, 30),
                        "wins": np.random.randint(5, 15),
                        "losses": np.random.randint(5, 15),
                        "no_result": 0
                    }
                    h2h_data[team]["no_result"] = h2h_data[team]["matches"] - h2h_data[team]["wins"] - h2h_data[team]["losses"]
            
            # Display performance data
            with team_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tournament Performance")
                    st.markdown(f"""
                    - **Matches Played:** {team_performance['matches_played']}
                    - **Wins:** {team_performance['wins']}
                    - **Losses:** {team_performance['losses']}
                    - **Points:** {team_performance['points']}
                    - **Net Run Rate:** {team_performance['net_run_rate']}
                    """)
                
                with col2:
                    st.subheader("Team Averages")
                    st.markdown(f"""
                    - **Average Score:** {team_performance['batting_avg']}
                    - **Average Conceded:** {team_performance['bowling_avg']}
                    - **Highest Score:** {team_performance['highest_score']}
                    - **Lowest Score:** {team_performance['lowest_score']}
                    """)
                
                # Performance chart
                st.subheader("Form in Last 5 Matches")
                match_results = ['W', 'L', 'W', 'W', 'L']
                
                # Display as colorful boxes
                html_str = '<div style="display: flex; gap: 10px;">'
                for result in match_results:
                    color = "#4CAF50" if result == 'W' else "#F44336"
                    html_str += f'<div style="width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; background-color: {color}; color: white; font-weight: bold; border-radius: 5px;">{result}</div>'
                html_str += '</div>'
                
                st.markdown(html_str, unsafe_allow_html=True)
            
            # Display player data
            with team_tabs[1]:
                st.subheader("Top Batsmen")
                batsmen = [p for p in players if p["role"] in ["Batsman", "All-rounder"]]
                batsmen_df = pd.DataFrame(batsmen)
                if "wickets" in batsmen_df.columns:
                    batsmen_df = batsmen_df.drop("wickets", axis=1)
                if "bowling_average" in batsmen_df.columns:
                    batsmen_df = batsmen_df.drop("bowling_average", axis=1)
                
                st.dataframe(batsmen_df, use_container_width=True)
                
                st.subheader("Top Bowlers")
                bowlers = [p for p in players if p["role"] in ["Bowler", "All-rounder"]]
                bowlers_df = pd.DataFrame(bowlers)
                if "batting_average" in bowlers_df.columns:
                    bowlers_df = bowlers_df.drop("batting_average", axis=1)
                
                st.dataframe(bowlers_df, use_container_width=True)
            
            # Display venue records
            with team_tabs[2]:
                st.subheader("Venue Performance")
                venue_df = pd.DataFrame(venue_records)
                
                st.dataframe(venue_df, use_container_width=True)
                
                # Create a bar chart for win percentage at venues
                for venue in venue_records:
                    venue["win_percentage"] = (venue["wins"] / venue["matches"]) * 100
                
                fig = px.bar(
                    venue_records, 
                    x="venue", 
                    y="win_percentage",
                    labels={"venue": "Venue", "win_percentage": "Win Percentage"},
                    title=f"{selected_team} Win Percentage by Venue",
                    color="win_percentage",
                    color_continuous_scale="RdYlGn"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display head-to-head data
            with team_tabs[3]:
                st.subheader("Head-to-Head Records")
                
                # Convert head-to-head data to DataFrame
                h2h_list = []
                for team, record in h2h_data.items():
                    h2h_list.append({
                        "opponent": team,
                        "matches": record["matches"],
                        "wins": record["wins"],
                        "losses": record["losses"],
                        "no_result": record["no_result"],
                        "win_percentage": (record["wins"] / record["matches"]) * 100
                    })
                
                h2h_df = pd.DataFrame(h2h_list)
                st.dataframe(h2h_df, use_container_width=True)
                
                # Create a bar chart for head-to-head win percentages
                fig = px.bar(
                    h2h_df, 
                    x="opponent", 
                    y="win_percentage",
                    labels={"opponent": "Opponent", "win_percentage": "Win Percentage"},
                    title=f"{selected_team} Win Percentage Against Opponents",
                    color="win_percentage",
                    color_continuous_scale="RdYlGn"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please wait for the data to load.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white;">
IPL 2025 Betting Analysis Assistant | For informational purposes only | Not financial advice
</div>
""", unsafe_allow_html=True)