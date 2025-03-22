import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_match_prediction_chart(prediction):
    """
    Create a visualization for match prediction results
    
    Args:
        prediction (dict): Prediction results with win probabilities
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the prediction chart
    """
    # Extract team names and probabilities
    labels = ['Team 1', 'Team 2', 'Draw/No Result']
    values = [
        prediction.get('team1_win_prob', 0), 
        prediction.get('team2_win_prob', 0), 
        prediction.get('draw_prob', 0)
    ]
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Match Outcome Probabilities",
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    # Add annotations
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        insidetextfont=dict(size=14)
    )
    
    return fig

def create_player_performance_chart(player_data):
    """
    Create a visualization for player performance
    
    Args:
        player_data (dict): Player statistics and performance data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the player performance chart
    """
    # Create radar chart for player performance
    categories = ['Batting', 'Bowling', 'Fielding', 'Experience', 'Form']
    
    # Extract player performance metrics (or generate sample data)
    values = [
        player_data.get('batting_score', np.random.randint(50, 100)),
        player_data.get('bowling_score', np.random.randint(50, 100)),
        player_data.get('fielding_score', np.random.randint(50, 100)),
        player_data.get('experience_score', np.random.randint(50, 100)),
        player_data.get('form_score', np.random.randint(50, 100))
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_data.get('name', 'Player')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=f"Performance Analysis: {player_data.get('name', 'Player')}",
        showlegend=False
    )
    
    return fig

def create_team_form_chart(team_data):
    """
    Create a visualization for team form over time
    
    Args:
        team_data (dict): Team performance data over time
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the team form chart
    """
    # Create line chart for team performance trend
    
    # Sample data if not provided
    if 'form_trend' not in team_data:
        dates = pd.date_range(start='2022-01-01', periods=10, freq='M')
        performance = np.random.randint(40, 90, size=10)
        
        form_trend = pd.DataFrame({
            'date': dates,
            'performance': performance
        })
    else:
        form_trend = team_data['form_trend']
    
    # Create line chart
    fig = px.line(
        form_trend, 
        x='date', 
        y='performance',
        title=f"Performance Trend: {team_data.get('name', 'Team')}",
        labels={'performance': 'Performance Score', 'date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Performance Score",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_odds_comparison_chart(match_data):
    """
    Create a visualization comparing betting odds with model predictions
    
    Args:
        match_data (dict): Match data with odds and predictions
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the odds comparison chart
    """
    # Extract team names
    team1 = match_data.get('team1', 'Team 1')
    team2 = match_data.get('team2', 'Team 2')
    
    # Extract probabilities
    model_probs = [
        match_data.get('team1_model_prob', 50),
        match_data.get('team2_model_prob', 50)
    ]
    
    odds_probs = [
        match_data.get('team1_odds_prob', 50),
        match_data.get('team2_odds_prob', 50)
    ]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='Model Prediction', x=[team1, team2], y=model_probs),
        go.Bar(name='Implied by Odds', x=[team1, team2], y=odds_probs)
    ])
    
    fig.update_layout(
        title="Model Predictions vs. Betting Odds",
        xaxis_title="Team",
        yaxis_title="Win Probability (%)",
        barmode='group',
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_head_to_head_chart(h2h_data):
    """
    Create a visualization for head-to-head record between teams
    
    Args:
        h2h_data (dict): Head-to-head statistics between two teams
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the head-to-head chart
    """
    # Extract team names
    team1 = h2h_data.get('team1', 'Team 1')
    team2 = h2h_data.get('team2', 'Team 2')
    
    # Extract win counts
    team1_wins = h2h_data.get('team1_wins', 0)
    team2_wins = h2h_data.get('team2_wins', 0)
    draws = h2h_data.get('draws', 0)
    
    # Create pie chart
    labels = [f"{team1} Wins", f"{team2} Wins", "Draws/No Results"]
    values = [team1_wins, team2_wins, draws]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
    )])
    
    # Update layout
    fig.update_layout(
        title_text=f"Head-to-Head: {team1} vs {team2}",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_feature_importance_chart(features_dict):
    """
    Create a visualization for feature importance in the prediction model
    
    Args:
        features_dict (dict): Dictionary of feature names and importance scores
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure for the feature importance chart
    """
    # Sort features by importance
    sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h'
    )])
    
    # Update layout
    fig.update_layout(
        title="Feature Importance in Prediction Model",
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=150)  # Add more left margin for feature names
    )
    
    return fig
