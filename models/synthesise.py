import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from preprocess import splitDate, splitTime
from bert_models import createLabelMapping, mapLabels

def generateDates(start, end, k):
    """
    Generate dates for each thread

    Parameters:

        start (datetime) : Start date to synthesise from 
        end (datetime) : End date to synthesise to
        k (int) : Number of dates to synthesise
    
    Returns:

        sorted(res) (list) : Sorted list of dates
    """
    res = []
    intdelta = int(datetime.timestamp(end) - datetime.timestamp(start))
    
    for _ in range(k):
        res.append(datetime.fromtimestamp(int(datetime.timestamp(start)) + random.randint(0, intdelta)))
    
    return sorted(res)

def synthesiseDates(df):
    """
    Synthesises dates for the dataset

    Parameters:

        df (pd.DataFrame) : The dataset

    Returns:

        new_df (pd.DataFrame) : Dataset with synthesised dates
    """
    # Get thread ids
    thread_ids = df['thread_id'].value_counts().index.tolist()

    # Create new dataframe to be returned
    new_df = pd.DataFrame()

    for tid in thread_ids:
        # Extract dataframes based on thread ids
        temp = df.loc[df['thread_id'] == tid]
        temp = temp.drop(temp.columns[0], axis = 1)
        # Generate new column of synthetic dates
        temp['synthetic_dates'] = generateDates(datetime(2023, 1, 1, 0, 0, 0),datetime(2023, 1, 31, 0, 0, 0), temp.shape[0])
        # Concatenate all thread_id based dataframes
        if new_df.empty:
            new_df = temp
        else:
            new_df = pd.concat([new_df, temp], axis = 0)
    
    new_df = new_df.sort_values(by = ['synthetic_dates'], ascending = True)
    return new_df

def getTimestamps(col):
    """
    Gets timestamps from dates

    Parameters:

        col (pd.Series) : Column of dates to transform

    Returns:
        
        ts (datetime) : Converted timestamp values
    """
    # Input a column of dates, return the timestamps
    ts = (pd.to_datetime(col, unit = 's')- pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return ts

def getTimestampOffsets(col):
    """
    Gets timestamp offsets from first date in dataset

    Parameters:

        col (pd.Series) : Date column

    Returns:

        offsetCol (pd.Series) : Column of offsets from first date in dataset
    """
    # Need to take in a col of timestamps
    refDate = col.iloc[0] # Set the first date as reference 
    offsetCol = col - refDate
    #offsetCol = pd.to_datetime(offsetCol)
    return offsetCol

def binariseDay(day_col):
    """
    Puts days into one-hot encoded form

    Parameters:

        day_col (pd.Series) : Column of days

    Returns:

        new_col (pd.Series) : Column of one-hot encoded days
    """
    new_col = []
    
    for i in day_col:
        new_arr = [0] * 31
        new_arr[int(i) - 1] = 1
        new_col.append(new_arr)

    return new_col

def binariseMonth(month_col):
    """
    Puts months into one-hot encoded form

    Parameters:

        month_col (pd.Series) : Column of months

    Returns:

        new_col (pd.Series) : Column of one-hot encoded months
    """
    new_col = []
    
    for i in month_col:
        new_arr = [0] * 12
        new_arr[int(i) - 1] = 1
        new_col.append(new_arr)
        
    return new_col

def binariseDayofWeek(dow_col):
    """
    Puts days of the week into one-hot encoded form

    Parameters:

        dow_col (pd.Series) : Column of days of the week

    Returns:

        new_col (pd.Series) : Column of one-hot encoded days of the week
    """
    new_col = []
    
    for i in dow_col:
        new_arr = [0] * 7
        new_arr[int(i) - 1] = 1
        new_col.append(new_arr)
        
    return new_col

def binariseHour(hour_col):
    """
    Puts hours into one-hot encoded form

    Parameters:

        hour_col (pd.Series) : Column of hours

    Returns:

        new_col (pd.Series) : Column of one-hot encoded hours
    """
    new_col = []
    
    for i in hour_col:
        new_arr = [0] * 24
        new_arr[int(i)] = 1
        new_col.append(new_arr)
        
    return new_col

def binariseYear(year_col):
    """
    Puts years into one-hot encoded form

    Parameters:

        year_col (pd.Series) : Column of years

    Returns:

        new_col (pd.Series) : Column of one-hot encoded years
    """
    new_col = []
    num_years = len(year_col.unique().tolist())
    mapping = dict()
    j = 0
    for i in year_col:
        if i not in mapping:
            mapping[i] = j
            j += 1

        new_arr = [0] * num_years
        new_arr[mapping[i]] = 1
        new_col.append(new_arr)
            
    return new_col

def main():
    df = pd.read_csv('../data/cleaned.csv')

    sorted_df = synthesiseDates(df)
    sorted_df['timestamps'] = getTimestamps(sorted_df['synthetic_dates'])
    sorted_df['offsets'] = getTimestampOffsets(sorted_df['timestamps'])

    sorted_df['synthesised_day_of_week_str'], sorted_df['synthesised_day_of_week_int'], sorted_df['synthesised_days'], sorted_df['synthesised_months'], sorted_df['synthesised_years'] = splitDate(sorted_df['synthetic_dates'])
    sorted_df['synthesised_hours'] = splitTime(sorted_df['synthetic_dates'])
    sorted_df['synthesised_day_binarised'] = binariseDay(sorted_df['synthesised_days'])
    sorted_df['synthesised_month_binarised'] = binariseMonth(sorted_df['synthesised_months'])
    sorted_df['synthesised_hour_binarised'] = binariseHour(sorted_df['synthesised_hours'])
    sorted_df['synthesised_dow_binarised'] = binariseDayofWeek(sorted_df['synthesised_day_of_week_int'])
    sorted_df['synthesised_year_binarised'] = binariseYear(sorted_df['synthesised_years'])
    #chosen_cols = ['id', 'creator_id', 'thread_id', 'content', 'tokens', 'created_on', 'binarised', 'all_cids', 'binarised_pids', 'all_pids', 'synthetic_dates', 'timestamps', 'offsets', 'synthesised_day_of_week_str', 'synthesised_day_of_week_int', 'synthesised_days', 'synthesised_months', 'synthesised_years', 'synthesised_hours']
    sorted_df.to_csv('../data/cleaned_and_synthesised.csv', index = False)

    threads = df['thread_id'].unique().tolist()
    new_list = sorted_df['thread_id'].tolist()
    mapping = createLabelMapping(threads)
    new_list = mapLabels(new_list, mapping)

    plt.scatter(range(len(new_list)), new_list, marker='x', color='blue')
    plt.xlabel('Message Index')
    plt.ylabel('Thread Number')
    plt.title('Message Distribution Across Threads in Synthesised Dataset')
    # plt.ylim(top=17)
    # plt.show()
    plt.savefig('../figures/synthesised_dist_old.png')

    # -- For synthesising discord data, uncomment if required --

    """df = pd.read_csv('../data/discord_test_hacking_cleaned.csv')
    start_date = datetime(2021, 1, 1, 0, 0, 0)  # Start date and time
    end_date = datetime(2022, 1, 30, 0, 0, 0)  # End date and time

    # Generate timestamps
    timestamps = []
    current_date = start_date
    for i in range(len(df)):
        timestamps.append(int(current_date.timestamp()))
        # Add a random time delta with an average of 5 minutes (300 seconds)
        current_date += timedelta(seconds=random.randint(240, 360))

    # Print the generated timestamps
    df['timestamps'] = timestamps

    df.to_csv('../data/discord_test_hacking_cleaned_and_synthesised.csv', index=False)"""


if __name__ == '__main__':
    main()

