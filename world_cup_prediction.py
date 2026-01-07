import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# --- App Configuration ---
st.set_page_config(page_title="World Cup AI", page_icon="⚽", layout="centered")

st.title("⚽ FIFA World Cup AI Predictor")
st.caption("Powered by XGBoost + Custom Elo Rating Engine")

# --- Elo Rating System ---
def expected_result(loc, opp):
    """Calculates expected score (probability of winning) based on Elo difference."""
    return 1 / (1 + 10 ** ((opp - loc) / 600))

def update_elo(current_elo, opponent_elo, actual_score, k_factor=32):
    """Updates Elo rating after a match."""
    expected = expected_result(current_elo, opponent_elo)
    new_elo = current_elo + k_factor * (actual_score - expected)
    return new_elo

# --- Data Loading & Feature Engineering ---
@st.cache_data
def load_and_process_data():
    """
    Loads match results, sorts by date, and calculates dynamic Elo ratings
    for every match in the dataset.
    """
    df = pd.read_csv('results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    current_elo = {}
    home_elos = []
    away_elos = []
    
    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        home_rating = current_elo.get(home, 1500)
        away_rating = current_elo.get(away, 1500)
        
        home_elos.append(home_rating)
        away_elos.append(away_rating)
        
        # Determine actual match outcome (1=Home Win, 0=Away Win, 0.5=Draw)
        if row['home_score'] > row['away_score']:
            home_actual, away_actual = 1, 0
        elif row['home_score'] < row['away_score']:
            home_actual, away_actual = 0, 1
        else:
            home_actual, away_actual = 0.5, 0.5
            
        # Higher K-factor for World Cup matches to increase sensitivity
        k = 60 if row['tournament'] == 'FIFA World Cup' else 30
        
        current_elo[home] = update_elo(home_rating, away_rating, home_actual, k)
        current_elo[away] = update_elo(away_rating, home_rating, away_actual, k)

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    
    return df, current_elo

try:
    df, current_elo = load_and_process_data()
except FileNotFoundError:
    st.error("❌ Error: 'results.csv' not found. Please ensure data file is in the root directory.")
    st.stop()

# --- Model Training ---
@st.cache_resource
def train_model(data):
    """
    Trains an XGBoost Classifier using Elo ratings and score differentials.
    Target: 0 (Home Win), 1 (Away Win), 2 (Draw).
    """
    def get_winner(row):
        if row['home_score'] > row['away_score']: return 0 
        elif row['away_score'] > row['home_score']: return 1
        else: return 2
    
    data['target'] = data.apply(get_winner, axis=1)
    data['elo_diff'] = data['home_elo'] - data['away_elo']
    
    X = data[['home_elo', 'away_elo', 'elo_diff']]
    y = data['target']
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    return model

model = train_model(df)
valid_teams = sorted(list(current_elo.keys()))

# --- Sidebar: Team Lookup ---
with st.sidebar:
    st.header("🕵️‍♂️ Team Name Checker")
    search = st.text_input("Search Team", "")
    if search:
        matches = [t for t in valid_teams if search.lower() in t.lower()]
        st.write(matches)
    else:
        st.write(valid_teams)

# --- Main Interface ---
tab1, tab2, tab3 = st.tabs(["⚔️ Head-to-Head", "🏆 2022 Replay", "🛠️ Custom Bracket"])

# Tab 1: Single Match Prediction
with tab1:
    st.header("AI Prediction (Elo-Based)")
    col1, col2 = st.columns(2)
    
    with col1:
        home = st.selectbox("Home Team", valid_teams, index=valid_teams.index("Brazil") if "Brazil" in valid_teams else 0)
        st.info(f"Elo: {int(current_elo.get(home, 1500))}")
    with col2:
        away = st.selectbox("Away Team", valid_teams, index=valid_teams.index("France") if "France" in valid_teams else 0)
        st.info(f"Elo: {int(current_elo.get(away, 1500))}")

    if st.button("Predict Match"):
        h_elo = current_elo[home]
        a_elo = current_elo[away]
        diff = h_elo - a_elo
        
        # Predict probabilities: Home, Away, Draw
        probs = model.predict_proba([[h_elo, a_elo, diff]])[0]
        
        st.subheader("Results")
        
        if probs[0] > probs[1] and probs[0] > probs[2]:
            st.success(f"🏆 {home} wins!")
        elif probs[1] > probs[0] and probs[1] > probs[2]:
            st.success(f"🏆 {away} wins!")
        else:
            st.warning("⚖️ Draw likely")
        
        st.write("Match Probabilities:")
        st.progress(float(probs[0]), text=f"{home} Win: {probs[0]:.1%}")
        st.progress(float(probs[2]), text=f"Draw: {probs[2]:.1%}")
        st.progress(float(probs[1]), text=f"{away} Win: {probs[1]:.1%}")

# Tab 2: 2022 Tournament Simulation
with tab2:
    st.header("🏆 2022 World Cup Replay")
    default_teams = [
        'Netherlands', 'United States', 'Argentina', 'Australia', 
        'Japan', 'Croatia', 'Brazil', 'South Korea',
        'England', 'Senegal', 'France', 'Poland',
        'Morocco', 'Spain', 'Portugal', 'Switzerland'
    ]
    
    if st.button("Run 2022 Simulation"):
        round_teams = default_teams
        while len(round_teams) > 1:
            st.markdown(f"### Round of {len(round_teams)}")
            next_round = []
            cols = st.columns(2)
            
            for i in range(0, len(round_teams), 2):
                t1 = round_teams[i]
                t2 = round_teams[i+1]
                e1 = current_elo[t1]
                e2 = current_elo[t2]
                
                probs = model.predict_proba([[e1, e2, e1-e2]])[0]
                
                # Advance winner based on higher win probability (ignores draws for knockouts)
                if probs[0] > probs[1]:
                    winner, win_prob = t1, probs[0]
                else:
                    winner, win_prob = t2, probs[1]
                
                next_round.append(winner)
                
                with cols[i%2]:
                    st.markdown(f"**{t1}** vs **{t2}**")
                    st.write(f"**Winner: {winner} ({int(win_prob*100)}%)**")
                    st.caption(f"{int(probs[0]*100)}% / {int(probs[2]*100)}% / {int(probs[1]*100)}%")
                    st.divider()
            
            round_teams = next_round
        
        st.balloons()
        st.success(f"🏆 Champion: {round_teams[0]}")

# Tab 3: Custom Scenario Builder
with tab3:
    st.header("🛠️ Build Your Own Bracket")
    st.write("Select 16 teams to create a custom Round of 16.")
    
    custom_teams = []
    
    for i in range(1, 9):
        c1, c2 = st.columns(2)
        with c1:
            default_idx = (i * 2) % len(valid_teams)
            t1 = st.selectbox(f"Match {i} - Team A", valid_teams, index=default_idx, key=f"m{i}_a")
        with c2:
            default_idx2 = (i * 2 + 1) % len(valid_teams)
            t2 = st.selectbox(f"Match {i} - Team B", valid_teams, index=default_idx2, key=f"m{i}_b")
        
        custom_teams.append(t1)
        custom_teams.append(t2)

    if st.button("Run Custom Simulation"):
        if len(set(custom_teams)) < 16:
            st.warning("⚠️ Warning: Duplicate teams selected.")
            
        round_teams = custom_teams
        while len(round_teams) > 1:
            st.markdown(f"### Round of {len(round_teams)}")
            next_round = []
            cols = st.columns(2)
            
            for i in range(0, len(round_teams), 2):
                t1 = round_teams[i]
                t2 = round_teams[i+1]
                e1 = current_elo.get(t1, 1500)
                e2 = current_elo.get(t2, 1500)
                
                probs = model.predict_proba([[e1, e2, e1-e2]])[0]
                
                if probs[0] > probs[1]:
                    winner, win_prob = t1, probs[0]
                else:
                    winner, win_prob = t2, probs[1]
                    
                next_round.append(winner)

                with cols[i%2]:
                    st.markdown(f"**{t1}** vs **{t2}**")
                    st.write(f"**Winner: {winner} ({int(win_prob*100)}%)**")
                    st.caption(f"{int(probs[0]*100)}% / {int(probs[2]*100)}% / {int(probs[1]*100)}%")
                    st.divider()
            
            round_teams = next_round
        
        st.balloons()
        st.success(f"🏆 Champion: {round_teams[0]}")