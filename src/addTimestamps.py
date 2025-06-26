import pandas as pd
import pytz
from datetime import datetime, timedelta

# Functions to fix time stamp issues
def add_timestamp_export(csv_path: str, tz_name: str = 'Asia/Bishkek') -> None:
    """
    Reads a CSV file with a 'Time' column in 'DD.MM.YYYY HH:MM:SS' format,
    localizes each datetime to the specified timezone, computes a 'UnixTimestamp'
    column (seconds since epoch), inserts it immediately after the 'Duration' column,
    and overwrites the original file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file to read and overwrite.
    tz_name : str, optional
        Timezone name (default: 'Asia/Bishkek').
    """
    df = pd.read_csv(csv_path)
    tz = pytz.timezone(tz_name)
    dt_series = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
    dt_series = dt_series.dt.tz_localize(tz)
    ts_series = (dt_series.astype('int64') // 10**6).astype(int)
    duration_idx = df.columns.get_loc('Duration') 
    df.insert(duration_idx + 1, 'UnixTimestamp', ts_series)

    df.to_csv(csv_path, index=False)

def add_timestamp_comments(csv_path: str, date_str: str, tz_name: str = 'Asia/Bishkek') -> None:
    """
    Reads a CSV file with a 'Time' column in 'HH:MM:SS' format (and other columns),
    combines each time with the provided date (YYYY-MM-DD), localizes to the specified
    timezone, computes a 'UnixTimestamp' column (seconds since epoch), inserts it 
    immediately after the 'Time' column, and overwrites the original file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file to read and overwrite.
    date_str : str
        Date string in 'YYYY-MM-DD' format to combine with each time.
    tz_name : str, optional
        Timezone name (default: 'Asia/Bishkek').
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    tz = pytz.timezone(tz_name)
    dt_series = pd.to_datetime(date_str + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    dt_series = dt_series.dt.tz_localize(tz)
    ts_series = (dt_series.astype('int64') // 10**9).astype(int)
    time_idx = df.columns.get_loc('Time')
    df.insert(time_idx + 1, 'UnixTimestamp', ts_series)
    df.to_csv(csv_path, index=False)

def add_timestamp_nasal(
    csv_path: str,
    start_datetime_str: str,
    tz_name: str = "Asia/Bishkek"
) -> None:
    """
    Reads a CSV whose first column is elapsed time in seconds (float or int) named arbitrarily,
    adds that many seconds onto the given start_datetime_str (YYYY-MM-DD HH:MM:SS),
    localizes to the specified timezone, computes Unix timestamps, and inserts them right
    after the elapsed‐seconds column. Finally, overwrites the CSV in place.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file to read and overwrite.
    start_datetime_str : str
        The starting point (date + time) in 'YYYY-MM-DD HH:MM:SS' format.
        This should already be in the Bishkek zone (UTC+6).
    tz_name : str, optional
        Timezone name (default: 'Asia/Bishkek').
    """
    df = pd.read_csv(csv_path)

    elapsed_col = df.columns[0]

    tz = pytz.timezone(tz_name)

    base_dt_naive = pd.to_datetime(start_datetime_str, format="%Y-%m-%d %H:%M:%S")
    base_dt = tz.localize(base_dt_naive)
    elapsed_td = pd.to_timedelta(df[elapsed_col], unit="ms")

    dt_series = base_dt + elapsed_td
    unix_series = (dt_series.view("int64") // 10**9).astype(int)
    insert_idx = 1 
    df.insert(insert_idx, "UnixTimestamp", unix_series)
    df.to_csv(csv_path, index=False)


def bishkek_to_unix(date_str, time_str, elapsed_seconds):
    """
    Convert Bishkek time to Unix timestamp in milliseconds.
    
    Args:
        date_str (str): Date in DD-MM-YYYY format
        time_str (str): Time in HH:MM:SS format
        elapsed_seconds (int/float): Elapsed seconds from the start time
    
    Returns:
        int: Unix timestamp in milliseconds (milliseconds since epoch)
    """
    # Parse the date and time
    datetime_str = f"{date_str} {time_str}"
    dt = datetime.strptime(datetime_str, "%d-%m-%Y %H:%M:%S")
    
    # Set timezone to Bishkek (UTC+6)
    bishkek_tz = pytz.timezone('Asia/Bishkek')
    dt_bishkek = bishkek_tz.localize(dt)
    
    # Add elapsed seconds
    final_dt = dt_bishkek + timedelta(seconds=elapsed_seconds)
    
    # Convert to Unix timestamp in milliseconds
    unix_timestamp = int(final_dt.timestamp() * 1000)
    
    return unix_timestamp

def add_unix_timestamps_to_csv(csv_file_path, date_str, time_str):
    """
    Read CSV file, add Unix timestamps as second column based on elapsed seconds in first column,
    and update the same file.
    
    Args:
        csv_file_path (str): Path to CSV file to modify
        date_str (str): Start date in DD-MM-YYYY format
        time_str (str): Start time in HH:MM:SS format
    
    Returns:
        pandas.DataFrame: Modified dataframe with Unix timestamps
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the first column (elapsed seconds)
    elapsed_seconds_col = df.iloc[:, 0]
    
    # Calculate Unix timestamps for each row
    unix_timestamps = [bishkek_to_unix(date_str, time_str, elapsed) for elapsed in elapsed_seconds_col]
    
    # Insert Unix timestamp (in milliseconds) as the second column
    df.insert(1, 'UnixTimestamp', unix_timestamps)
    
    # Save back to the same file
    df.to_csv(csv_file_path, index=False)
    
    print(f"Modified CSV file: {csv_file_path}")
    print(f"Added {len(unix_timestamps)} Unix timestamps")
    
    return df
