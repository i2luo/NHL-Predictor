import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl
from io import StringIO

# List of NHL teams with their abbreviations
teams = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG"
}

# Define the Excel file path
file_path = "nhl_gamelog.xlsx"
sheet_name = "Sheet1"

# Create a blank DataFrame to store all teams' data
all_games = pd.DataFrame()

for team_name, team_abbr in teams.items():
    print(f"Scraping {team_name}...")

    # Construct the URL
    url = f"https://www.hockey-reference.com/teams/{team_abbr}/2025_gamelog.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table
    table = soup.find(name='table')

    if table:
        # Remove all rows with class "thead"
        for row in table.find_all('tr', class_='thead'):
            row.decompose()

        # Convert table to string and parse with pandas
        table_html = StringIO(str(table))
        df = pd.read_html(table_html)[0]

        # Flatten multi-index columns if applicable
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        # Find the column name that corresponds to "Score Rslt" (7th column)
        # Column names might vary based on how pandas reads the HTML table
        result_column = df.columns[6]  # Assuming it's the 7th column (index 6)
        
        # Filter to include only rows with a value in the result column
        # This keeps only games that have been played
        df = df[df[result_column].notna()]

        # Add a new column specifying the team name
        df.insert(0, "Team", team_name)

        # Append to the all_games DataFrame
        all_games = pd.concat([all_games, df], ignore_index=True)

# Save all data to an Excel file
all_games.to_excel(file_path, index=False, sheet_name=sheet_name)

# Load the Excel file to format headers
wb = openpyxl.load_workbook(file_path)
ws = wb.active

# Define proper headers
header_values = [
    "Team", "RK", "GTM", "Date", "Home", "Opponents", "Results", "GF", "GA", "OT", "TEAM SOG", "TEAM PIM", "TEAM PPG", 
    "TEAM PPO", "TEAM SHG", "OPPONENT SOG", "OPPONENT PIM", "OPPONENT PPG", "OPPONENT PPO", "OPPONENT SHG", 
    "FOW", "FOL", "FO%", "CF", "CA", "CF%", "FF", "FA", "FF%", "oZS%", "PDO"
]
# Apply headers in the first row
for col, header in enumerate(header_values, start=1):
    ws.cell(row=1, column=col, value=header)

# Save the formatted file
wb.save(file_path)

print(f"Excel file '{file_path}' successfully created with all completed games for all 32 teams!")
