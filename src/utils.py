import fastf1
import pandas as pd
import datetime
fastf1.Cache.enable_cache('./.cache') 

def convert(n):
    try:
        n = int(n)
    except:
        n = float(n)
    return str(datetime.timedelta(seconds = n))

team_driver_performance = pd.read_csv('.cache/hist_data/team_driver_performance.csv')
race_int = team_mapping = {'Alfa Romeo': 1, 'AlphaTauri': 2, 'Alpine': 3, 'Renault':3,'Aston Martin': 4, 'Ferrari': 5, 'Haas F1 Team': 1, 'Kick Sauber': 6, 'McLaren': 7, 'Mercedes': 8, 'RB': 2, 'Racing Bulls': 2, 'Red Bull Racing': 9, 'Williams': 10}
MODEL_PATH='./model/model.joblib'

prac_columns = ['DriverNumber_prac','LapTime_prac', 'Sector1Time_prac', 'Sector2Time_prac', 'Sector3Time_prac', 'SpeedI1_prac', 'SpeedI2_prac', 'TyreLife_prac','Position_prac','Driver_prac']
old_qual_columns = ['LapTime_old_qual', 'Position_old_qual','Sector1Time_old_qual', 'Sector2Time_old_qual', 'Sector3Time_old_qual','Driver_old_qual']

driver_numbers = {
    "HAM": 44,   # Hamilton
    "RUS": 63,   # Russell
    "LEC": 16,   # Leclerc
    "PIA": 81,   # Piastri
    "NOR": 4,    # Norris
    "VER": 1,    # Verstappen
    "SAI": 55,   # Sainz
    "ALB": 23,   # Albon
    "HUL": 27,   # Hulkenberg
    "ALO": 14,   # Alonso
    "TSU": 22,   # Tsunoda
    "GAS": 10,   # Gasly
    "STR": 18,   # Stroll
    "OCO": 31,   # Ocon
    "COL": 43,   # Colapinto
    # Added missing ones:
    "ANT": 12,   # Antonelli :contentReference[oaicite:0]{index=0}
    "BEA": 87,   # Bearman :contentReference[oaicite:1]{index=1}
    "LAW": 30,   # Lawson (But his code is “LAW”; however, as surname Lawson → “LAW”) “LAW”:30 :contentReference[oaicite:2]{index=2}
    "BOR": 5,    # Bortoleto :contentReference[oaicite:3]{index=3}
    "HAD": 6     # Hadjar :contentReference[oaicite:4]{index=4}
}

reverse_driver_numbers = {v: k for k, v in driver_numbers.items()}

def get_lap_data(session):
    laps = session.laps
    laps = laps.pick_accurate().reset_index(drop=True)

    # Fix boolean indexing by wrapping conditions in parentheses
    mask = ((laps['IsAccurate'] == True) & 
            (laps['Deleted'] == False) & 
            (laps['IsPersonalBest'] == True))
    laps = laps[mask].copy()  # Create explicit copy to avoid SettingWithCopyWarning

    # Remove duplicate 'Position' from drop list
    laps.drop(columns=['LapStartTime',
        'LapStartDate', 'TrackStatus', 'Position', 'Deleted', 'DeletedReason',
        'FastF1Generated', 'IsAccurate', 'SpeedFL', 'SpeedST', 'Sector1SessionTime',
        'Sector2SessionTime', 'Sector3SessionTime', 'PitOutTime', 'PitInTime'],
        inplace=True)

    # Convert timedelta columns to seconds
    for col in laps.select_dtypes(include='timedelta64[ns]').columns:
        laps[col] = laps[col].dt.total_seconds()

    # Add session information
    laps['session'] = session.name
    laps['country'] = session.event.Country
    laps['year'] = session.event.year
    laps['Date'] = session.event.EventDate

    # Get fastest lap per driver and reset index
    laps = (laps.groupby('Driver', as_index=False)
            .apply(lambda x: x.loc[x['LapTime'].idxmin()])
            .sort_values('LapTime'))
    laps.reset_index(inplace=True)
    laps.rename(columns={'index':'Position'}, inplace=True)
    laps['Position'] = laps['Position'] + 1
    return laps

def load_data(race,year,mode='train'):
    no_prac = False
    no_qual = False
    e_name = ''
    # Load sessions 
    try:
        session = fastf1.get_session(year, race, 'Q')
        e_name = session.event.EventName
        session.load()
        lap_qual = get_lap_data(session)
    except Exception as e:
        print(f"Error loading qualifying session for {race} in {year}: {e}")
        lap_qual = None

    try:
        session_old_qual = fastf1.get_session(year-1,race,'Q')
        e_name = session_old_qual.event.EventName
        session_old_qual.load()
        lap_old_qual = get_lap_data(session_old_qual)
        lap_old_qual.columns = [f'{col}_old_qual' if col not in ['Team'] else col for col in lap_old_qual.columns]
    except Exception as e:
        print(f"Error loading old qualifying session for {race} in {year-1}: {e}")
        lap_old_qual = pd.DataFrame([[-1]*len(old_qual_columns)]*20,columns=old_qual_columns)  # Empty DataFrame if old qualifying session fails
        lap_old_qual['Driver_old_qual'] = lap_qual['Driver']
        no_qual = True

    try:
        try:
            session_p = fastf1.get_session(year,race,'FP3')
            session_p.load()
            e_name = session_p.event.EventName
            lap_prac = get_lap_data(session_p)
            lap_prac.columns = [f'{col}_prac' if col not in ['Team'] else col for col in lap_prac.columns]
        except:
            session_p = fastf1.get_session(year,race,'SQ')
            session_p.load()
            e_name = session_p.event.EventName
            lap_prac = get_lap_data(session_p)
            lap_prac.columns = [f'{col}_prac' if col not in ['Team'] else col for col in lap_prac.columns]
    except Exception as e:
        print(f"Error loading practice session for {race} in {year}: {e}")
        lap_prac = pd.DataFrame([[-1]*len(prac_columns)]*20,columns=prac_columns)  # Empty DataFrame if practice session fails
        lap_prac['DriverNumber_prac'] = lap_qual['DriverNumber']
        lap_prac['Driver_prac'] = lap_qual['Driver']
        no_prac = True

    if no_prac and no_qual:
        return None, None, None
    else:
        lap_old_qual_clean = lap_old_qual[old_qual_columns]
        lap_prac_clean = lap_prac[prac_columns]
        tdp = team_driver_performance[team_driver_performance['Country'] == e_name]
        if len(tdp) == 0:
            tdp = pd.DataFrame([[-1]*len(tdp.columns)]*20,columns=tdp.columns)
            print(f"Team Driver Performance data missing for {e_name}, filling with -1s.")
            final_lap = (
                lap_prac_clean
                .merge(lap_old_qual_clean, left_on='Driver_prac', right_on='Driver_old_qual')
                .drop(columns=['Driver_prac', 'Driver_old_qual'])
                .set_index('DriverNumber_prac')
            )
            for col in tdp.columns:
                if col not in final_lap.columns and col != 'Name':
                    final_lap[col] = -1
        else:
            final_lap = (
                lap_prac_clean
                .merge(lap_old_qual_clean, left_on='Driver_prac', right_on='Driver_old_qual')
                .merge(
                    tdp,
                    left_on='Driver_prac', right_on='Name'
                )
                .drop(columns=['Driver_prac', 'Driver_old_qual','Name','Country'])
                .set_index('DriverNumber_prac')
            )
        final_lap['race_event'] = [race]*len(final_lap) if type(race) == int else [int(session.event.RoundNumber)]*len(final_lap)
        
        if lap_qual is not None:
            y = lap_qual[['LapTime', 'Position', 'DriverNumber']].set_index('DriverNumber').loc[final_lap.index]
            y_laptime = y.pop('LapTime').to_list()
            y_position = y.pop('Position').to_list()
        else:
            y_laptime = [-1]*len(final_lap)
            y_position = [-1]*len(final_lap)

        if mode == 'train':
            return final_lap.values, y_laptime, y_position
        elif mode == 'predict':
            return final_lap.values, list(map(lambda x : reverse_driver_numbers.get(int(x),'Unknown'),final_lap.index.to_list()))
        else:
            raise ValueError("Mode should be either 'train' or 'predict'")
    
def load_data_to_predict(race,year):
    X, drivers = load_data(race,year,mode='predict')
    return pd.DataFrame(X).fillna(-1).to_numpy(), drivers if X is not None else None