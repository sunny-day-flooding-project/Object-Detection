import os
import requests
import pytz
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def fetch_data(start_date, end_date, sensor_id, authorization):
    """
        Calculate the product of two numbers.

        Parameters
        ----------
        start_date:

        end_date:

        sensor_id : str
            The sensor identifier.
        authorization: tuple
            API username and password as a tuple of strings.

        Returns
        -------
        json_data
            API datastream for chosen sensor over the specified times.
    """

    # Sunny Day Flooding Project water level API
    base_url = "https://api-sunnydayflood.apps.cloudapps.unc.edu/get_water_level"

    # Parameters necessary for requesting data from the API
    query_params = {
        "min_date": start_date.strftime("%Y-%m-%d"),
        "max_date": end_date.strftime("%Y-%m-%d"),
        "sensor_ID": sensor_id
    }

    # Received data from the API
    response = requests.get(base_url, params=query_params, auth=authorization)

    # If data pulled successfully, return the data
    if response.status_code == 200:
        json_data = response.json()
        return json_data
    return None


def sensor_list_generator(location_name):
    """
        Provide a sensor identity list for a given location.

        Parameters
        ----------
        location_name : str
            Name of sensor location.

        Returns
        -------
        sensor_ids
            List of sensor ids as strings.
    """

    if location_name.lower() == 'carolina beach':
        sensor_ids = ["CB_01", "CB_02", "CB_03"]
    elif location_name.lower() == 'down east':
        sensor_ids = ["DE_01", "DE_02", "DE_03"]
    elif location_name.lower() == 'new bern':
        sensor_ids = ["NB_01", "NB_02"]
    elif location_name.lower() == 'beaufort':
        sensor_ids = ["BF_01"]
    else:
        raise ValueError("Location name not recognized.")
    return sensor_ids


def data_retrieval_input_val(total_min_date, total_max_date, authorization, sensor_ids):
    """
        Input validation for get_sunnyd_data function.

        Parameters
        ----------
        total_min_date : datetime
            Start date.
        total_max_date : datetime
            End date.
        authorization : tuple
            API username and password as tuple of strings.
        sensor_ids : list
            A list of strings representing sensor identifiers.

        Returns
        -------
        None
            This function raises value errors if a value was not input.
    """

    if total_min_date is None or total_max_date is None:
        raise ValueError("Please provide total_min_date and total_max_date.")
    if authorization is None:
        raise ValueError("Please provide API authorization details.")
    if sensor_ids is None:
        raise ValueError("Please provide either location name or list of desired sensor IDs.")


def get_sunnyd_data(authorization, sensor_ids=None, total_min_date=None, total_max_date=None):
    """
        This function retrieves data for the requested inputs and returns a dataframe.

        Parameters
        ----------
        authorization : tuple
            API username and password as tuple of strings.
        sensor_ids : str or list
            A list of strings representing sensor identifiers.
        total_min_date : datetime or timedelta
            Start date.
        total_max_date : datetime or timedelta
            End date.

        Returns
        -------
        combined_data
            A dataframe from the water level API for provided sensor IDs over the requested time window.
    """

    # Input validation
    data_retrieval_input_val(total_min_date, total_max_date, authorization, sensor_ids)

    # Convert sensor location to list of IDs
    if type(sensor_ids) == str:
        sensor_ids = sensor_list_generator(sensor_ids)

    # Calculate the total number of days to fetch data for
    total_days = (total_max_date - total_min_date).days
    sunnyd_data = []

    print(f"Downloading data from the SunnyD API for {sensor_ids}... ")

    # Using ThreadPoolExecutor for asynchronous requests
    # Do not increase the number of workers, this results in data gaps in the dataframe
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Creating a list of futures for each day's data retrieval for each sensor
        futures = [
            executor.submit(fetch_data, total_min_date + timedelta(days=i),
                            total_min_date + timedelta(days=i + 1), sensor_id, authorization)
            for sensor_id in sensor_ids
            for i in range(total_days)
        ]

        # Iterating through each future to retrieve data and process it
        for future in futures:
            json_data = future.result()  # Get the result from each future (asynchronous task)
            if json_data:
                df = pd.DataFrame(json_data)
                df['date'] = pd.to_datetime(df['date'], utc=True)
                df['date_surveyed'] = pd.to_datetime(df['date_surveyed'], utc=True)
                sunnyd_data.append(df)

    print("Here is your data, human. Sincerely, 🤖")

    # Concatenate all retrieved DataFrames and sort by the 'sensor_ID' and 'date' columns
    sunnyd_data = [df for df in sunnyd_data if not df.empty]

    if sunnyd_data:
        combined_data = pd.concat(sunnyd_data, ignore_index=True)
        combined_data.sort_values(by=['date'], inplace=True)
        combined_data.reset_index(drop=True, inplace=True)
        return combined_data
    return None


def reassign_abbr_flood_numbers(df):
    """
        This function reassigns the flood event numbers for the abbreviated flood event CSV.

        Parameters
        ----------
        df : dataframe
            Dataframe of abbreviated flood events.

        Returns
        -------
        df
            The same dataframe with reassigned flood event numbers.
    """

    df = df.copy()

    # Convert 'start_time_UTC' and 'end_time_UTC' to Timestamp
    df['start_time_UTC'] = pd.to_datetime(df['start_time_UTC'], utc=True)
    df['end_time_UTC'] = pd.to_datetime(df['end_time_UTC'], utc=True)

    # Sort the dataframe by start_time_UTC
    df = df.sort_values(by=['start_time_UTC'])

    # Initialize variables to keep track of the current flood event number and end time
    current_event_number = 1
    current_end_time = df.iloc[0]['end_time_UTC']

    # Iterate through the rows of the dataframe
    for index, row in df.iterrows():
        # Check if the current event overlaps with the previous one
        if row['start_time_UTC'] < current_end_time:
            # Assign the same flood event number as the previous event
            df.loc[index, 'flood_event'] = current_event_number
            # Update the current end time if the current event's end time is greater
            current_end_time = max(current_end_time, row['end_time_UTC'])
        else:
            # Assign a new flood event number
            current_event_number += 1
            df.loc[index, 'flood_event'] = current_event_number
            # Update the current end time
            current_end_time = row['end_time_UTC']

    # Ensure 'flood_event' column has int dtype
    df['flood_event'] = df['flood_event'].astype(int)

    return df


def gen_abbr_flood_event_csv(dataframe, csv_filename='abbr_flood_events.csv'):
    """
        This function generates an abbreviated flood event CSV file.

        Parameters
        ----------
        dataframe : dataframe
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.

        Returns
        -------
        None
            This function has no return statement but saves a CSV with the specified name.
    """

    # Read the existing CSV file if it exists
    try:
        existing_data = pd.read_csv(csv_filename)
        last_event_number = existing_data['flood_event'].max() + 1 if 'flood_event' in existing_data else 0
    except FileNotFoundError:
        column_names = ['flood_event', 'sensor_ID', 'start_time_UTC', 'end_time_UTC',
                        'start_time_EST', 'end_time_EST', 'duration_(hours)',
                        'max_road_water_level_(ft)', 'max_road_water_level_(m)']
        existing_data = pd.DataFrame(columns=column_names)
        last_event_number = 0

    # Initialize variables to track flood events
    flood_start_time = None
    max_water_level = 0
    flood_events = []

    eastern = pytz.timezone('EST')

    dataframe = dataframe.sort_values(by=['date'])
    sensors = dataframe['sensor_ID'].unique().tolist()

    for sensor in sensors:

        filtered_dataframe = dataframe[dataframe['sensor_ID'] == sensor]

        for index, row in filtered_dataframe.iterrows():
            water_level = row['road_water_level_adj']
            timestamp = row['date']

            # Check for the start of a new flood event
            if water_level > 0.02 and flood_start_time is None:
                flood_start_time = timestamp
                max_water_level = water_level

            # Track maximum water level during the flood event
            if water_level > max_water_level:
                max_water_level = water_level

            # Check for the end of a flood event
            if water_level < 0 and flood_start_time is not None:
                flood_end_time = timestamp

                # Convert start and end times to Eastern Time (EST)
                start_time_est = flood_start_time.astimezone(eastern)
                end_time_est = flood_end_time.astimezone(eastern)

                # Save flood event details
                flood_events.append({
                    'flood_event': last_event_number,
                    'sensor_ID': sensor,
                    'start_time_UTC': flood_start_time,
                    'end_time_UTC': flood_end_time,
                    'start_time_EST': start_time_est,
                    'end_time_EST': end_time_est,
                    'duration_(hours)': round((end_time_est - start_time_est).total_seconds() / 3600, 3),
                    'max_road_water_level_(ft)': round(max_water_level, 3),
                    'max_road_water_level_(m)': round(max_water_level / 3.28, 3)
                })

                # Reset variables for the next flood event
                last_event_number += 1
                flood_start_time = None
                max_water_level = 0

    # Create a DataFrame from the collected flood event details
    flood_event_df = pd.DataFrame(flood_events)

    if existing_data.empty:
        # Reassign flood event numbers
        updated_data = reassign_abbr_flood_numbers(flood_event_df)

    else:
        # Concatenate existing data with new flood event data
        merged_df = pd.concat([existing_data, flood_event_df], ignore_index=True)
        merged_df['start_time_UTC'] = merged_df['start_time_UTC'].astype(str)
        filtered_df = merged_df[~merged_df.duplicated(subset=['sensor_ID', 'start_time_UTC'], keep='first')]
        updated_data = reassign_abbr_flood_numbers(filtered_df)

    # Save the updated DataFrame to the CSV file
    updated_data.to_csv(csv_filename, index=False)
    # flood_event_df.to_csv(csv_filename, index=False)


def reassign_flood_numbers(df):
    """
        This function reassigns the flood event numbers for the flood event CSV.

        Parameters
        ----------
        df : dataframe
            Dataframe of flood events.

        Returns
        -------
        df
            The same dataframe with reassigned flood event numbers.
    """

    df = df.copy()

    # Convert 'start_time_UTC' and 'end_time_UTC' to Timestamp
    df['start_time_UTC'] = pd.to_datetime(df['start_time_UTC'], utc=True)
    df['end_time_UTC'] = pd.to_datetime(df['end_time_UTC'], utc=True)
    df['time_UTC'] = pd.to_datetime(df['time_UTC'], utc=True)

    # Create a dictionary to map alphanumeric sensor IDs to numeric values
    sensor_id_mapping = {'CB_01': 1, 'CB_02': 2, 'CB_03': 3, 'BF_01': 4,
                         'DE_01': 5, 'DE_02': 6, 'DE_03': 7, 'NB_01': 8, 'NB_02': 9}
    # Add a new column with numeric representations of Sensor_ID
    df['Sensor_ID_numeric'] = df['sensor_ID'].map(sensor_id_mapping)

    # Sort the dataframe by start_time_UTC
    df = df.sort_values(by=['time_UTC', 'Sensor_ID_numeric'])

    # Reset index to reflect the sorted order
    df = df.reset_index(drop=True)

    # Drop the intermediate column used for sorting
    df = df.drop(columns=['Sensor_ID_numeric'])

    # Dataframe of rows indicating the end of flood events
    complete_rows = df[df['duration_(hours)'].notnull()]

    # Initialize flood numbers and assigned event dictionary.
    flood_event_number = 1
    last_assigned_event = {}
    current_end_time = complete_rows.iloc[0]['end_time_UTC']

    # Iterate over each complete row (end of flood event)
    for index, row in complete_rows.iterrows():
        # Check if the row starts before the current end time
        if row['start_time_UTC'] < current_end_time:

            # Check if the sensor is not in the assigned events
            if row['sensor_ID'] not in last_assigned_event:
                # Set the flood event number and add the index and sensor ID to the dictionary
                df.loc[(df.index <= index) & (df['sensor_ID'] == row['sensor_ID']), 'flood_event'] = flood_event_number
                last_assigned_event[row['sensor_ID']] = index

            # If the sensor ID is in the event dictionary
            elif row['sensor_ID'] in last_assigned_event:
                # Check if the current index is higher than the last dictionary entry
                # Set the flood event number and update the dictionary
                df.loc[(df.index > last_assigned_event[row['sensor_ID']]) &
                       (df.index <= index) & (df['sensor_ID'] == row['sensor_ID']), 'flood_event'] = flood_event_number
                last_assigned_event[row['sensor_ID']] = index

            # Update the current end time if the current event's end time is greater
            current_end_time = max(current_end_time, row['end_time_UTC'])
        else:
            # Increase flood event number
            flood_event_number += 1

            # Check if the sensor is not in the assigned events
            if row['sensor_ID'] not in last_assigned_event:
                df.loc[(df.index <= index) & (df['sensor_ID'] == row['sensor_ID']), 'flood_event'] = flood_event_number
                last_assigned_event[row['sensor_ID']] = index

            # If the sensor ID is in the event dictionary
            elif row['sensor_ID'] in last_assigned_event:
                # Check if the current index is higher than the last dictionary entry
                # Set the flood event number and update the dictionary
                df.loc[(df.index > last_assigned_event[row['sensor_ID']]) &
                       (df.index <= index) & (df['sensor_ID'] == row['sensor_ID']), 'flood_event'] = flood_event_number
                last_assigned_event[row['sensor_ID']] = index

            # Update the current end time
            current_end_time = row['end_time_UTC']

    return df


def gen_flood_tracker(dataframe, csv_filename='flood_events.csv'):
    """
        This function generates a flood event CSV file with every datapoint.

        Parameters
        ----------
        dataframe : dataframe
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.

        Returns
        -------
        None
            This function has no return statement but saves a CSV with the specified name.
    """

    # Read the existing CSV file if it exists
    try:
        existing_data = pd.read_csv(csv_filename)
        last_event_number = existing_data['flood_event'].max() + 1 if 'flood_event' in existing_data else 0
    except FileNotFoundError:
        column_names = ['flood_event', 'sensor_ID', 'start_time_UTC',
                        'start_time_EST', 'road_water_level_adj']
        existing_data = pd.DataFrame(columns=column_names)
        last_event_number = 0

    # Initialize variables to track flood events
    flood_start_time = None
    flood_end_time = None
    max_water_level = 0
    flood_events = []

    eastern = pytz.timezone('EST')

    dataframe = dataframe.sort_values(by=['date'])
    sensors = dataframe['sensor_ID'].unique().tolist()

    for sensor in sensors:

        filtered_dataframe = dataframe[dataframe['sensor_ID'] == sensor]

        for index, row in filtered_dataframe.iterrows():
            water_level = row['road_water_level_adj']
            timestamp = row['date']

            # Check for the start of a new flood event
            if water_level > 0.02 and flood_start_time is None:
                flood_start_time = timestamp
                max_water_level = water_level

            # Track maximum water level during the flood event
            if water_level > max_water_level:
                max_water_level = water_level

            # Check for the end of a flood event
            if water_level < 0 and flood_start_time is not None:
                flood_end_time = timestamp

                # Convert start and end times to Eastern Time (EST)
                start_time_est = flood_start_time.astimezone(eastern)
                end_time_est = flood_end_time.astimezone(eastern)

            if flood_start_time is not None and flood_end_time is None:
                # Save flood event details
                flood_events.append({
                    'flood_event': last_event_number,
                    'sensor_ID': sensor,
                    'time_UTC': timestamp,
                    'time_EST': timestamp.astimezone(eastern),
                    'water_level': water_level,
                    'start_time_UTC': None,
                    'end_time_UTC': None,
                    'start_time_EST': None,
                    'end_time_EST': None,
                    'duration_(hours)': None,
                    'max_road_water_level_(ft)': None,
                    'max_road_water_level_(m)': None
                })
            elif flood_start_time is not None and flood_end_time is not None:
                # Save flood event details and add additional details if it is the end of an event
                flood_events.append({
                    'flood_event': last_event_number,
                    'sensor_ID': sensor,
                    'time_UTC': timestamp,
                    'time_EST': timestamp.astimezone(eastern),
                    'water_level': water_level,
                    'start_time_UTC': flood_start_time,
                    'end_time_UTC': flood_end_time,
                    'start_time_EST': start_time_est,
                    'end_time_EST': end_time_est,
                    'duration_(hours)': round((end_time_est - start_time_est).total_seconds() / 3600, 3),
                    'max_road_water_level_(ft)': round(max_water_level, 3),
                    'max_road_water_level_(m)': round(max_water_level / 3.28, 3)
                })

                # Reset variables for the next flood event
                last_event_number += 1
                flood_start_time = None
                flood_end_time = None
                max_water_level = 0

    # Create a DataFrame from the collected flood event details
    flood_event_df = pd.DataFrame(flood_events)

    if existing_data.empty:
        # Reassign flood numbers
        updated_data = reassign_flood_numbers(flood_event_df)

    else:
        # Concatenate existing data with new flood event data
        merged_df = pd.concat([existing_data, flood_event_df], ignore_index=True)
        merged_df['time_UTC'] = merged_df['time_UTC'].astype(str)
        filtered_df = merged_df[~merged_df.duplicated(subset=['sensor_ID', 'time_UTC'], keep='first')]
        updated_data = reassign_flood_numbers(filtered_df)

    # Save the updated DataFrame to the CSV file
    updated_data.to_csv(csv_filename, index=False)


def find_outages(dataframe, csv_filename='sensor_outages.csv'):
    """
        This function generates a list of outages in a CSV file and returns a dataframe of outages.

        Parameters
        ----------
        dataframe : dataframe
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.

        Returns
        -------
        outage_dataframe
            This function returns the outages as a dataframe and saves a CSV of outages.
    """

    # List the sensor IDs present in the dataframe
    sensor_id_list = dataframe['sensor_ID'].unique().tolist()

    # Initialize list of outages
    outages = []
    for sensor_id in sensor_id_list:

        # Filter and sort the dataframe
        filtered_dataframe = dataframe[dataframe['sensor_ID'] == sensor_id]
        sorted_dataframe = filtered_dataframe.sort_values(by='date')
        sorted_dataframe = sorted_dataframe.reset_index(drop=True)

        # Generate dataframe of outages (entries where the gap between datapoints was larger than an hour)
        time_differences = sorted_dataframe['date'].diff()
        gaps_larger_than_hour = sorted_dataframe[time_differences > pd.Timedelta(hours=1)]

        # List the indices of the dataframe where the outages exist
        indices_of_gaps = gaps_larger_than_hour.index

        for i in indices_of_gaps:
            # Record the start and end time of the outage
            outage_start_time = sorted_dataframe['date'][i - 1]
            outage_end_time = sorted_dataframe['date'][i]

            eastern = pytz.timezone('EST')

            start_time_est = outage_start_time.astimezone(eastern)
            end_time_est = outage_end_time.astimezone(eastern)

            outages.append({
                'outage_number': None,
                'sensor_ID': sensor_id,
                'start_time_UTC': outage_start_time,
                'end_time_UTC': outage_end_time,
                'start_time_EST': start_time_est,
                'end_time_EST': end_time_est,
                'duration_(hours)': round((end_time_est - start_time_est).total_seconds() / 3600, 3),
            })

    # Convert the outage list to a dataframe and sort by the start time
    outage_dataframe = pd.DataFrame(outages)
    outage_dataframe = outage_dataframe.sort_values(by='start_time_UTC')

    # Save the outages as a CSV
    outage_dataframe.to_csv(csv_filename, index=False)

    return outage_dataframe


def check_for_outage_during_flood(outage_csv='sensor_outages.csv', abbr_flood_csv='abbr_flood_events.csv'):
    """
        This function compares the flood events to outages and determines if there were outages during floods.

        Parameters
        ----------
        outage_csv : str, optional
            A string for the outages CSV filename.
        abbr_flood_csv : str, optional
            A string for the abbreviated flood events CSV filename.

        Returns
        -------
        None
            This function has no return statement but reassigns outage numbers and adds indicators of outages during
            flood events to the outage and flood events CSVs.
    """

    # Read the CSVs in as dataframes
    abbr_floods = pd.read_csv(abbr_flood_csv)
    outage_dataframe = pd.read_csv(outage_csv)

    # Create columns for flags in the two dataframes
    outage_dataframe['during_flood_event'] = None
    outage_dataframe['flood_event_number'] = None
    abbr_floods['outage'] = None

    # Initialize outage numbers
    outage_number = 1

    for index2, row2 in outage_dataframe.iterrows():

        # Assign and increment the outage number
        outage_dataframe.at[index2, 'outage_number'] = outage_number
        outage_number += 1

        # Pull start time and sensor ID for the current outage
        start_time_to_check = row2['start_time_UTC']
        sensor_id = row2['sensor_ID']
        for index1, row1 in abbr_floods.iterrows():
            # If the sensor IDs match and the outage is within the flood event time boundaries
            if row1['start_time_UTC'] <= start_time_to_check <= row1['end_time_UTC'] and row1['sensor_ID'] == sensor_id:
                # Add flags to both dataframes
                outage_dataframe.at[index2, 'during_flood_event'] = 'Yes'
                outage_dataframe.at[index2, 'flood_event_number'] = row1['flood_event']
                abbr_floods.at[index1, 'outage'] = 'Yes'
                break  # Break the inner loop once a match is found

    # Save the updated dataframes back to CSVs
    abbr_floods.to_csv(abbr_flood_csv, index=False)
    outage_dataframe.to_csv(outage_csv, index=False)


def plot_and_save_flood_plots(sunnyd_data, csv_filename='abbr_flood_events.csv'):
    """
        This function plots the flood events listed in the abbreviated flood event CSV file.

        Parameters
        ----------
        sunnyd_data : dataframe
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            Filename of the abbreviated flood events CSV file.

        Returns
        -------
        None
            This function has no return statement but saves PNGs for every flood event to a folder ('flood_plots').
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Note: This plotting function will not rewrite plots in place of existing plots if the
    # 'pull_data_gen_csvs_and_plots' function is called retroactively to fill csvs for prior events and may create
    # empty plots as well.
    # The function uses the abbreviated flood events spreadsheet and dataframe from the API. So, in the retroactive
    # case between existing events there is no data in the dataframe for events outside the dates of the API call.
    # This generates empty plots for already plotted events. If the API call is prior to all plotted events, no plots
    # will be overwritten because of the logic used to check for existing events and numbers.
    # ------------------------------------------------------------------------------------------------------------------

    # Create a folder to save the plots if it doesn't exist
    folder_name = 'flood_plots'
    if os.path.exists(folder_name):
        print(f'Folder {folder_name} already exists.')
    elif not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder {folder_name} created successfully.')

    # Read the abbreviated flood events to a dataframe
    abbr_dataframe = pd.read_csv(csv_filename)

    # Iterate over each row in the DataFrame
    for index, row in abbr_dataframe.iterrows():
        sensor_id = row['sensor_ID']
        event_number = row['flood_event']

        flood_start_time = pd.to_datetime(row['start_time_UTC'], utc=True)
        flood_end_time = pd.to_datetime(row['end_time_UTC'], utc=True)

        plot_start_time = flood_start_time - timedelta(days=1)
        plot_end_time = flood_end_time + timedelta(days=1)

        # Filter the DataFrame based on the time range and sensor ID
        filtered_df = sunnyd_data[(sunnyd_data['date'] >= plot_start_time) &
                                  (sunnyd_data['date'] <= plot_end_time) &
                                  (sunnyd_data['sensor_ID'] == sensor_id)]

        # Create a plot
        plt.figure(figsize=(12, 10))

        # Plot the water level with a solid line
        plt.plot(filtered_df['date'], filtered_df['road_water_level_adj'],
                 label='Water Level', linestyle='-', color='#427e93')

        # Plot the roadway with a solid line
        plt.plot(filtered_df['date'], filtered_df['road_elevation'] - filtered_df['road_elevation'],
                 label='Roadway', linestyle='-', linewidth=2, color='black')

        # Identify the indices within the flood event timeframe
        flood_indices = filtered_df[(filtered_df['date'] >= flood_start_time) &
                                    (filtered_df['date'] <= flood_end_time)].index

        # Plot markers for the water level within the flood event timeframe
        plt.plot(filtered_df.loc[flood_indices, 'date'],
                 filtered_df.loc[flood_indices, 'road_water_level_adj'],
                 marker='x', markersize=5, linestyle='None', color='#cc0000', label='Flood Event Data Points')

        # Set the y-axis limits
        plt.ylim(-2.5, 2)

        # Set ticks
        plt.xticks(rotation=45)
        plt.yticks(np.arange(-2.5, 2.25, 0.25))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.25))

        # Add gridlines
        plt.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray')
        plt.grid(which='minor', axis='y', linestyle='-', linewidth='0.5', color='black')

        # Create axis labels and title
        plt.xlabel('Date')
        plt.ylabel('Water Level (ft above road)')
        plt.title(f'Flood Event {event_number} for {sensor_id}')

        # Add a legend
        plt.legend(loc='upper right')

        # Save the plot as a PNG file in the folder
        # filename = os.path.join(folder_name, f"flood_event_{event_number}_{sensor_id}_.png")
        # plt.savefig(filename)

        # Save the plot as a PNG file in the folder
        filename = os.path.join(folder_name, f"flood_event_{event_number}_{sensor_id}.png")
        if not os.path.exists(filename):  # Check if the file already exists
            plt.savefig(filename)
            # print(f"Plot saved: {filename}")
        else:
            print(f"Plot already exists: {filename}")

        # Close the plot to release memory
        plt.close()


def pull_data_gen_csvs_and_plots(authorization, location, start_date, end_date):
    """
        This function generates an abbreviated flood event CSV file.

        Parameters
        ----------
        authorization : tuple
            Sunny Day Flooding Project water level API username and password.
        location : str or list
            A string for the location of the sensor or a list of sensor IDs as strings.
        start_date : datetime
            Start date.
        end_date : datetime
            End date.

        Returns
        -------
        download_data
            This function returns the dataframe of water level data from the Sunny Day Flooding Project API and
            generates a flood event CSV, abbreviated flood event CSV, sensor outage CSV, and a folder of flood event
            plots as PNGs.
    """

    download_data = get_sunnyd_data(authorization, location, start_date, end_date)
    print('Generating CSVs...')
    try:
        gen_flood_tracker(download_data)
        gen_abbr_flood_event_csv(download_data)
        print('CSV files created successfully.')
    except Exception as e:
        print(f"An error occurred: {e}")

    print('Checking for sensor outages...')
    try:
        find_outages(download_data)
        check_for_outage_during_flood()
        print('Sensor outage logs generated and CSVs appended.')
    except Exception as e:
        print(f"An error occurred: {e}")

    print('Plotting flood events...')
    try:
        plot_and_save_flood_plots(download_data)
        print('Plotting completed.')
    except Exception as e:
        print(f"An error occurred: {e}")

    return download_data


def num_of_flood_days_by_start(csv_name='abbr_flood_events.csv'):
    """
        This function returns a count of flood days only including the start date in UTC.

        Parameters
        ----------
        csv_name : str, optional
            The name of the csv containing the list of abbreviated events for counting the days.

        Returns
        -------
        None
            This function prints the number of flooding days to the command window.
    """

    # Read dataframe
    read_df = pd.read_csv(csv_name)

    # Convert 'start_time_UTC' to Pandas datetime object
    read_df['start_time_UTC'] = pd.to_datetime(read_df['start_time_UTC'])

    # Extract date part from datetime column
    read_df['date'] = read_df['start_time_UTC'].dt.date

    # Count the number of unique dates
    num_days = read_df['date'].nunique()

    return print("Number of days of flooding:", num_days)


def num_of_flood_days(timezone='EST', csv_name='abbr_flood_events.csv'):
    """
        This function generates a count of flood events by calendar days including both start and end dates in
        EST or UTC timezones.

        Parameters
        ----------
        timezone : str
            The timezone, either EST or UTC to count dates by.
        csv_name : str, optional
            The name of the csv containing the list of abbreviated events for counting the days.

        Returns
        -------
        None
            This function prints the number of flooding days to the command window.
    """

    # Read dataframe
    read_df = pd.read_csv(csv_name)

    if timezone == 'UTC':

        # Convert 'start_time_UTC' to Pandas datetime object
        read_df['start_time'] = pd.to_datetime(read_df['start_time_UTC'])
        read_df['end_time'] = pd.to_datetime(read_df['end_time_UTC'])

    if timezone == 'EST':

        # Convert 'start_time_EST' to Pandas datetime object
        read_df['start_time'] = pd.to_datetime(read_df['start_time_EST'])
        read_df['end_time'] = pd.to_datetime(read_df['end_time_EST'])

    # Extract date part from datetime column
    read_df['start_date'] = read_df['start_time'].dt.date
    read_df['end_date'] = read_df['end_time'].dt.date

    # Initialize an empty list to store all dates
    all_dates = []

    # Iterate over each row and append all dates in the range
    for index, row in read_df.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        date_range = pd.date_range(start=start_date, end=end_date)
        all_dates.extend(date_range)

    # Convert the list of dates to a Pandas Series
    all_dates = pd.Series(all_dates, name='combined_date')

    num_days = all_dates.nunique()

    return print("Number of days of flooding:", num_days)
