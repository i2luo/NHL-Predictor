import pandas as pd
import numpy as np
import datetime
import os

# Dictionary to map team abbreviations to full names
TEAM_ABBR_TO_NAME = {
    'VAN': 'Vancouver Canucks',
    'EDM': 'Edmonton Oilers',
    'CGY': 'Calgary Flames',
    'SEA': 'Seattle Kraken',
    'VGK': 'Vegas Golden Knights',
    'LAK': 'Los Angeles Kings',
    'SJS': 'San Jose Sharks',
    'ANA': 'Anaheim Ducks',
    'COL': 'Colorado Avalanche',
    'MIN': 'Minnesota Wild',
    'DAL': 'Dallas Stars',
    'WPG': 'Winnipeg Jets',
    'STL': 'St. Louis Blues',
    'NSH': 'Nashville Predators',
    'CHI': 'Chicago Blackhawks',
    'ARI': 'Arizona Coyotes',
    'TOR': 'Toronto Maple Leafs',
    'BOS': 'Boston Bruins',
    'TBL': 'Tampa Bay Lightning',
    'FLA': 'Florida Panthers',
    'DET': 'Detroit Red Wings',
    'OTT': 'Ottawa Senators',
    'MTL': 'Montreal Canadiens',
    'BUF': 'Buffalo Sabres',
    'NYR': 'New York Rangers',
    'NYI': 'New York Islanders',
    'NJD': 'New Jersey Devils',
    'PIT': 'Pittsburgh Penguins',
    'PHI': 'Philadelphia Flyers',
    'WSH': 'Washington Capitals',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
}

# Function to load the schedule data from Excel
def load_schedule(file_path):
    """
    Load the schedule data from Excel file
    Column 1: Date
    Column 2: Home Team
    Column 3: Away Team
    """
    try:
        schedule_df = pd.read_excel(file_path)
        
        # Rename columns if they don't have proper names
        if schedule_df.columns[0] == 0 or 'Unnamed' in str(schedule_df.columns[0]):
            schedule_df.columns = ['Date', 'Home_Team', 'Away_Team']
        
        # Convert date strings to datetime objects if needed
        if not pd.api.types.is_datetime64_any_dtype(schedule_df['Date']):
            schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
        
        # Print first few rows for debugging
        print("Schedule data (first 5 rows):")
        print(schedule_df.head())
        
        # Add a column to identify if Canucks are home or away
        schedule_df['Canucks_Home'] = schedule_df.apply(
            lambda row: 'VAN' in str(row['Home_Team']) or 'Vancouver' in str(row['Home_Team']), 
            axis=1
        )
        
        # Convert team abbreviations to full names
        schedule_df['Home_Team_Full'] = schedule_df['Home_Team'].apply(
            lambda x: TEAM_ABBR_TO_NAME.get(x, x) if isinstance(x, str) else x
        )
        
        schedule_df['Away_Team_Full'] = schedule_df['Away_Team'].apply(
            lambda x: TEAM_ABBR_TO_NAME.get(x, x) if isinstance(x, str) else x
        )
        
        return schedule_df
    
    except Exception as e:
        print(f"Error loading schedule: {e}")
        print("File structure:")
        try:
            temp_df = pd.read_excel(file_path)
            print(temp_df.head())
            print(f"Columns: {temp_df.columns.tolist()}")
        except:
            print("Unable to read file structure")
        raise

# Function to load team statistics
def load_team_stats():
    """
    In a real implementation, this would load statistics from an API or database.
    For this example, we'll create synthetic data for NHL teams.
    """
    # NHL teams
    teams = [
        'Vancouver Canucks', 'Edmonton Oilers', 'Calgary Flames', 'Seattle Kraken',
        'Vegas Golden Knights', 'Los Angeles Kings', 'San Jose Sharks', 'Anaheim Ducks',
        'Colorado Avalanche', 'Minnesota Wild', 'Dallas Stars', 'Winnipeg Jets',
        'St. Louis Blues', 'Nashville Predators', 'Chicago Blackhawks', 'Arizona Coyotes',
        'Toronto Maple Leafs', 'Boston Bruins', 'Tampa Bay Lightning', 'Florida Panthers',
        'Detroit Red Wings', 'Ottawa Senators', 'Montreal Canadiens', 'Buffalo Sabres',
        'New York Rangers', 'New York Islanders', 'New Jersey Devils', 'Pittsburgh Penguins',
        'Philadelphia Flyers', 'Washington Capitals', 'Carolina Hurricanes', 'Columbus Blue Jackets'
    ]
    
    # Create a dictionary for team statistics
    team_stats = {}
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Define mean values for different team tiers
    top_tier = {'GF': 3.4, 'GA': 2.6, 'SOG': 33, 'SOG_against': 28, 'PPG': 1.1, 'PIM': 8, 'CF': 53, 'FF': 52, 'FO': 52, 'PDO': 101, 'OT_WIN_PCT': 0.62}
    mid_tier = {'GF': 3.0, 'GA': 3.0, 'SOG': 30, 'SOG_against': 30, 'PPG': 0.8, 'PIM': 9, 'CF': 50, 'FF': 50, 'FO': 50, 'PDO': 100, 'OT_WIN_PCT': 0.5}
    low_tier = {'GF': 2.6, 'GA': 3.4, 'SOG': 28, 'SOG_against': 33, 'PPG': 0.5, 'PIM': 10, 'CF': 47, 'FF': 48, 'FO': 48, 'PDO': 99, 'OT_WIN_PCT': 0.38}
    
    # Define team tiers for 2025 season (with Canucks as a good team for this example)
    top_teams = ['Vancouver Canucks', 'Edmonton Oilers', 'Colorado Avalanche', 'Vegas Golden Knights', 
                'Toronto Maple Leafs', 'Boston Bruins', 'Tampa Bay Lightning', 'Carolina Hurricanes', 
                'New York Rangers', 'Dallas Stars', 'Florida Panthers']
    
    bottom_teams = ['Chicago Blackhawks', 'San Jose Sharks', 'Anaheim Ducks', 'Montreal Canadiens', 
                   'Philadelphia Flyers', 'Columbus Blue Jackets', 'Buffalo Sabres']
    
    # Generate stats for each team
    for team in teams:
        if team in top_teams:
            base = top_tier
            tier_factor = 1.2
        elif team in bottom_teams:
            base = low_tier
            tier_factor = 0.8
        else:
            base = mid_tier
            tier_factor = 1.0
        
        # Add some random variation to make teams unique
        team_stats[team] = {
            'Rolling_GF': base['GF'] * (0.9 + 0.2 * np.random.random()),
            'Rolling_GA': base['GA'] * (0.9 + 0.2 * np.random.random()),
            'Rolling_TEAM_SOG': base['SOG'] * (0.9 + 0.2 * np.random.random()),
            'Rolling_OPPONENT_SOG': base['SOG_against'] * (0.9 + 0.2 * np.random.random()),
            'Rolling_TEAM_PPG': base['PPG'] * (0.8 + 0.4 * np.random.random()),
            'Rolling_TEAM_PIM': base['PIM'] * (0.8 + 0.4 * np.random.random()),
            'Rolling_CFPct': base['CF'] * (0.95 + 0.1 * np.random.random()),
            'Rolling_FFPct': base['FF'] * (0.95 + 0.1 * np.random.random()),
            'Rolling_FOPct': base['FO'] * (0.95 + 0.1 * np.random.random()),
            'Rolling_PDO': base['PDO'] * (0.99 + 0.02 * np.random.random()),
            'Team_Strength': tier_factor * (0.9 + 0.2 * np.random.random()),
            'OT_Win_Pct': base['OT_WIN_PCT'] * (0.9 + 0.2 * np.random.random())  # Added OT win percentage
        }
    
    # Add abbreviations as keys too
    abbr_stats = {}
    for abbr, name in TEAM_ABBR_TO_NAME.items():
        if name in team_stats:
            abbr_stats[abbr] = team_stats[name]
    
    # Merge dictionaries
    team_stats.update(abbr_stats)
    
    return team_stats

# Function to create feature vector for prediction
def create_feature_vector(home_team, away_team, team_stats, days_since_season_start):
    """
    Create the feature vector for the prediction model
    """
    # Try to get team stats, handling both full names and abbreviations
    try:
        home_stats = team_stats[home_team]
    except KeyError:
        # Try to convert from abbreviation to full name
        home_full = TEAM_ABBR_TO_NAME.get(home_team, home_team)
        try:
            home_stats = team_stats[home_full]
        except KeyError:
            print(f"Warning: Could not find stats for team: {home_team} or {home_full}")
            print(f"Available teams: {list(team_stats.keys())[:5]}...")
            # Use default stats
            home_stats = {
                'Team_Strength': 1.0,
                'Rolling_GF': 3.0,
                'Rolling_GA': 3.0,
                'Rolling_TEAM_SOG': 30,
                'Rolling_OPPONENT_SOG': 30,
                'Rolling_TEAM_PPG': 0.8,
                'Rolling_CFPct': 50,
                'Rolling_PDO': 100,
                'OT_Win_Pct': 0.5
            }
    
    try:
        away_stats = team_stats[away_team]
    except KeyError:
        # Try to convert from abbreviation to full name
        away_full = TEAM_ABBR_TO_NAME.get(away_team, away_team)
        try:
            away_stats = team_stats[away_full]
        except KeyError:
            print(f"Warning: Could not find stats for team: {away_team} or {away_full}")
            print(f"Available teams: {list(team_stats.keys())[:5]}...")
            # Use default stats
            away_stats = {
                'Team_Strength': 1.0,
                'Rolling_GF': 3.0,
                'Rolling_GA': 3.0,
                'Rolling_TEAM_SOG': 30,
                'Rolling_OPPONENT_SOG': 30,
                'Rolling_TEAM_PPG': 0.8,
                'Rolling_CFPct': 50,
                'Rolling_PDO': 100,
                'OT_Win_Pct': 0.5
            }
    
    # Create feature vector
    features = {
        'Days_Since_Season_Start': days_since_season_start,
        'Home_Team_Strength': home_stats['Team_Strength'],
        'Away_Team_Strength': away_stats['Team_Strength'],
        'Home_Rolling_GF': home_stats['Rolling_GF'],
        'Home_Rolling_GA': home_stats['Rolling_GA'],
        'Away_Rolling_GF': away_stats['Rolling_GF'],
        'Away_Rolling_GA': away_stats['Rolling_GA'],
        'Home_Rolling_SOG': home_stats['Rolling_TEAM_SOG'],
        'Home_Rolling_SOG_Against': home_stats['Rolling_OPPONENT_SOG'],
        'Away_Rolling_SOG': away_stats['Rolling_TEAM_SOG'],
        'Away_Rolling_SOG_Against': away_stats['Rolling_OPPONENT_SOG'],
        'Home_Rolling_PPG': home_stats['Rolling_TEAM_PPG'],
        'Away_Rolling_PPG': away_stats['Rolling_TEAM_PPG'],
        'Home_Rolling_CF': home_stats['Rolling_CFPct'],
        'Away_Rolling_CF': away_stats['Rolling_CFPct'],
        'Home_Rolling_PDO': home_stats['Rolling_PDO'],
        'Away_Rolling_PDO': away_stats['Rolling_PDO'],
        'Home_OT_Win_Pct': home_stats.get('OT_Win_Pct', 0.5),
        'Away_OT_Win_Pct': away_stats.get('OT_Win_Pct', 0.5)
    }
    
    return features

# Predict score using Poisson distribution with team statistics and resolve ties
def predict_score(features):
    """
    Predict scores based on team stats using a modified Poisson model
    Includes overtime resolution for tied games
    """
    # Base expected goals (league average is around 3.0 per team)
    base_home_expected = 3.1  # Home teams score slightly more on average
    base_away_expected = 2.9
    
    # Apply modifiers based on team strengths and stats
    home_expected = base_home_expected * (
        features['Home_Team_Strength'] * 0.5 + 
        (features['Home_Rolling_GF'] / features['Away_Rolling_GA']) * 0.3 +
        (features['Home_Rolling_SOG'] / features['Away_Rolling_SOG_Against']) * 0.1 +
        (features['Home_Rolling_CF'] / 50) * 0.05 +
        (features['Home_Rolling_PDO'] / 100) * 0.05
    )
    
    away_expected = base_away_expected * (
        features['Away_Team_Strength'] * 0.5 + 
        (features['Away_Rolling_GF'] / features['Home_Rolling_GA']) * 0.3 +
        (features['Away_Rolling_SOG'] / features['Home_Rolling_SOG_Against']) * 0.1 +
        (features['Away_Rolling_CF'] / 50) * 0.05 +
        (features['Away_Rolling_PDO'] / 100) * 0.05
    )
    
    # Add small random variation
    home_expected *= (0.9 + 0.2 * np.random.random())
    away_expected *= (0.9 + 0.2 * np.random.random())
    
    # Generate scores using Poisson distribution
    home_score = np.random.poisson(home_expected)
    away_score = np.random.poisson(away_expected)
    
    # Check if the game is tied after regulation
    is_overtime = False
    if home_score == away_score:
        is_overtime = True
        # Determine overtime winner based on OT win percentages
        home_ot_win_pct = features['Home_OT_Win_Pct']
        away_ot_win_pct = features['Away_OT_Win_Pct']
        
        # Normalize to make sure probabilities sum to 1
        total = home_ot_win_pct + away_ot_win_pct
        home_ot_win_pct /= total
        away_ot_win_pct /= total
        
        # Determine winner
        if np.random.random() < home_ot_win_pct:
            home_score += 1  # Home team wins in OT
        else:
            away_score += 1  # Away team wins in OT
    
    return home_score, away_score, is_overtime

# Main function to predict all games
def predict_canucks_games(schedule_file):
    """
    Predict all Vancouver Canucks games in the schedule
    """
    # Load the schedule
    schedule = load_schedule(schedule_file)
    
    # Load team statistics
    team_stats = load_team_stats()
    
    # Get current date
    current_date = datetime.datetime.now()
    
    # Assume season started on October 1, 2024
    season_start = datetime.datetime(2024, 10, 1)
    
    # Create results dataframe
    results = []
    
    # Process each game
    for idx, game in schedule.iterrows():
        # Skip games that have already happened
        if game['Date'] < current_date:
            continue
        
        # Calculate days since season start
        days_since_start = (game['Date'] - season_start).days
        
        # Determine home and away teams
        if game['Canucks_Home']:
            home_team = game['Home_Team']
            away_team = game['Away_Team']
        else:
            home_team = game['Home_Team']
            away_team = game['Away_Team']  # 'VAN' should be one of these
        
        # Create feature vector
        features = create_feature_vector(home_team, away_team, team_stats, days_since_start)
        
        # Predict the score
        home_score, away_score, is_overtime = predict_score(features)
        
        # Determine if Vancouver wins
        vancouver_is_home = game['Canucks_Home']
        vancouver_score = home_score if vancouver_is_home else away_score
        opponent_score = away_score if vancouver_is_home else home_score
        vancouver_win = vancouver_score > opponent_score
        
        # Add to results
        results.append({
            'Date': game['Date'],
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Home_Score': home_score,
            'Away_Score': away_score,
            'Vancouver_Score': vancouver_score,
            'Opponent_Score': opponent_score,
            'Vancouver_Win': vancouver_win,
            'Is_Overtime': is_overtime
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add win probability
    results_df['Win_Probability'] = results_df.apply(
        lambda row: calculate_win_probability(row['Home_Score'], row['Away_Score'], row['Home_Team'] == 'VAN' or row['Home_Team'] == 'Vancouver Canucks'),
        axis=1
    )
    
    # Format results with OT indicator
    results_df['Predicted_Score'] = results_df.apply(
        lambda row: f"{row['Home_Team']} {row['Home_Score']} - {row['Away_Score']} {row['Away_Team']}{' (OT)' if row['Is_Overtime'] else ''}",
        axis=1
    )
    
    return results_df

# Function to calculate win probability
def calculate_win_probability(home_score, away_score, canucks_home):
    """
    Calculate win probability based on predicted scores
    """
    diff = abs(home_score - away_score)
    
    if (home_score > away_score and canucks_home) or (away_score > home_score and not canucks_home):
        # Canucks win
        if diff == 1:
            # Close game
            return min(0.5 + 0.15, 0.95)
        else:
            # Larger margin
            return min(0.5 + 0.1 * diff, 0.95)
    else:
        # Canucks lose
        if diff == 1:
            # Close game
            return max(0.5 - 0.15, 0.05)
        else:
            # Larger margin
            return max(0.5 - 0.1 * diff, 0.05)

# Run predictions and save to file
def main(schedule_file):
    """
    Main function to run predictions and save results
    """
    try:
        # Run predictions
        results = predict_canucks_games(schedule_file)
        
        # Calculate overall statistics
        total_games = len(results)
        
        if total_games == 0:
            print("No future games found in the schedule.")
            return pd.DataFrame()
            
        wins = results['Vancouver_Win'].sum()
        losses = total_games - wins
        win_percentage = wins / total_games
        
        ot_games = results['Is_Overtime'].sum()
        ot_percentage = (ot_games / total_games) * 100
        
        regulation_wins = sum((results['Vancouver_Win']) & (~results['Is_Overtime']))
        ot_wins = sum((results['Vancouver_Win']) & (results['Is_Overtime']))
        
        avg_goals_for = results['Vancouver_Score'].mean()
        avg_goals_against = results['Opponent_Score'].mean()
        
        # Print summary
        print(f"\nVancouver Canucks Prediction Summary")
        print(f"==================================")
        print(f"Predicted Record: {wins:.0f}-{losses:.0f}")
        print(f"Regulation Wins: {regulation_wins:.0f}")
        print(f"Overtime/Shootout Wins: {ot_wins:.0f}")
        print(f"Win Percentage: {win_percentage:.3f}")
        print(f"Games Going to Overtime: {ot_games:.0f} ({ot_percentage:.1f}%)")
        print(f"Average Goals For: {avg_goals_for:.2f}")
        print(f"Average Goals Against: {avg_goals_against:.2f}")
        print(f"==================================\n")
        
        # Save results to Excel
        output_file = 'vancouver_canucks_predictions.xlsx'
        results.to_excel(output_file, index=False)
        print(f"Detailed predictions saved to {output_file}")
        
        return results
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    # Replace with your actual file path
    schedule_file = "future_games.xlsx"
    
    if os.path.exists(schedule_file):
        predictions = main(schedule_file)
        
        if not predictions.empty:
            # Display the next 5 games
            print("\nUpcoming 10 Games Predictions:")
            for idx, game in predictions.head(10).iterrows():
                ot_indicator = " (OT)" if game['Is_Overtime'] else ""
                print(f"{game['Date'].strftime('%Y-%m-%d')}: {game['Home_Team']} {game['Home_Score']} - {game['Away_Score']} {game['Away_Team']}{ot_indicator} (Win Prob: {game['Win_Probability']:.2f})")
    else:
        print(f"Error: Schedule file '{schedule_file}' not found.")
        print("Please provide a valid Excel file with Canucks schedule.")
        print("Format required: Column 1 = Date, Column 2 = Home Team, Column 3 = Away Team")
