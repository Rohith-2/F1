import time
import fastf1
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

fastf1.Cache.enable_cache('./.cache')  # replace with your cache directory
fastf1.Cache.offline_mode(True)
years = list(range(2018, 2026))

def get_lap_data(session):
    """
    Get lap data for a given session.
    Args:
        session (fastf1.core.Session): The session object from FastF1.
    Returns:
        pd.DataFrame: A DataFrame containing lap data with relevant columns.
    """
    lap = session.laps
    lap_prac = lap[(lap['IsAccurate']==True) & (lap['IsPersonalBest']==True)]
    lap_prac = lap_prac.groupby('Driver').apply(
        lambda x: x[x['LapTime'] == x['LapTime'].min()]).sort_values('LapTime').reset_index(drop=True)[['Driver','Team']]
    for col in lap_prac.select_dtypes(include='timedelta64[ns]').columns:
        lap_prac[col] = lap_prac[col].dt.total_seconds()
    lap_prac.reset_index(inplace=True)
    lap_prac.rename(columns={'index':'Position'},inplace=True)
    lap_prac['Position'] = lap_prac['Position'] + 1
    lap_prac['session'] = session.name
    lap_prac['country'] = session.event.Country
    lap_prac['year'] = session.event.year
    lap_prac['Date'] = session.event.EventDate
    
    return lap_prac

def fetch_lap_data(year, race, race_type):
    try:
        session = fastf1.get_session(year, race, race_type)
        session.load()
        time.sleep(5)  # To avoid overwhelming the server

        return get_lap_data(session)
    except Exception as e:
        print(f"Could not load session for year: {year} and race {race} due to {e}")
        return None
    
if __name__ == "__main__":

    qualifying_sessions = pd.DataFrame([])
    race_type = 'Q'
    tasks = [(year, race, race_type) for race in range(1, 25) for year in years]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_lap_data, year, race, race_type) for year, race, race_type in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                qualifying_sessions = pd.concat([qualifying_sessions, result], ignore_index=True)

    current_drivers = qualifying_sessions[qualifying_sessions['year']==years[-1]].Driver.unique().tolist()
    print(f"Current Drivers: {current_drivers}")
    circuits = qualifying_sessions['country'].unique().tolist()
    print(f"Circuits: {circuits}")
    team_mapping = dict(zip(sorted(qualifying_sessions.Team.value_counts().keys().tolist()),[1,2,3,4,5,1,6,7,8,2,9,10,11]))
    print(f"Team Mapping: {team_mapping}")

    driver_stats_list = []

    for i in current_drivers:

        driver_sessions = qualifying_sessions[qualifying_sessions['Driver'] == i]
        recent_sessions = driver_sessions[driver_sessions['year'] == years[-1]]
        avg_driver_pos = driver_sessions.groupby('country')['Position'].mean().sort_values().to_dict()
        team_name = driver_sessions['Team'].values[0]
        team_perf = qualifying_sessions[qualifying_sessions['Team'] == team_name].groupby('country')['Position'].mean().sort_values().to_dict()
        
        try : 
            lp = int(recent_sessions[recent_sessions['country'] == con]['Position'].iloc[0])
        except:
            lp = -1
        
        try:
            tf = float(team_perf[con])
        except:
            tf = -1
        
        if 'Great Britain' in avg_driver_pos.keys() and 'United Kingdom' in avg_driver_pos.keys():
            avg_driver_pos['Great Britain'] = (avg_driver_pos['United Kingdom'] + avg_driver_pos['Great Britain'])/2
            avg_driver_pos.pop('United Kingdom')
        elif 'United Kingdom' in avg_driver_pos.keys() and  'Great Britain' not in avg_driver_pos.keys():
            avg_driver_pos['Great Britain'] = avg_driver_pos['United Kingdom']
            avg_driver_pos.pop('United Kingdom')
        
        for con in avg_driver_pos.keys():
            driver_stats_list.append({
                'Name': i,
                'Avg_Position_Past': float(driver_sessions['Position'].mean()),
                'Best_Position_Past': int(driver_sessions['Position'].min()),
                'Worst_Position_Past': int(driver_sessions['Position'].max()),
                'Best_Position_Recent': int(recent_sessions['Position'].min()),
                'Worst_Position_Recent': int(recent_sessions['Position'].max()),
                'Avg_Position_Recent': float(recent_sessions['Position'].mean()),
                'Last_Position': lp,
                'Avg_Finish_in_this_circuit' : float(avg_driver_pos[con]),
                'Avg_Team_Finish_in_this_circuit' : tf,
                'Country': con,
                'Team': team_mapping[team_name]
            })

    pd.DataFrame(driver_stats_list).to_csv('.cache/hist_data/team_driver_performance.csv',index=False)