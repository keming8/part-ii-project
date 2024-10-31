import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from preprocess import splitTime, splitDate
from synthesise import getTimestampOffsets
from bert_models import *


def interleave_helper(prev, curr, interleave_rate):
    """
    Helper function to interleave threads.

    Parameters:

        prev (list) : Previous thread.
        curr (list) : Current thread.
        interleave_rate (float) : Percentage, from the back, of the prev to interleave.

    Returns:

        new_thread_df (pd.DataFrame) : New interleaving of threads
    """
    new_thread = []
    j = 0
    k = 0
    prev_prob = 0.8

    if len(curr) < interleave_rate * len(prev):
        interleave_rate = len(curr) / len(prev)

    while j < len(prev):
        # Aim to only interleave last interleave_rate % of prev
        if len(prev) - j > interleave_rate * len(prev):
            new_thread.append(prev.iloc[j])
            j += 1
        else: 
            choice = np.random.choice([0, 1], p=[prev_prob, 1-prev_prob])
            if choice == 0:
                new_thread.append(prev.iloc[j])
                j += 1
            else:
                if k < len(curr):
                    new_thread.append(curr.iloc[k])
                    k += 1
                    prev_prob = min(prev_prob + 0.01, 1)
        
    while k < len(curr):
        new_thread.append(curr.iloc[k])
        k += 1
    
    new_thread_df = pd.DataFrame(new_thread)
    return new_thread_df

def new_synthesiseDates(n):
    """
    Phase 2 method of synthesising dates.

    Parameters:

        n (int) : Number of messages to get dates for

    Returns:

        timestamp_row (list(int)) : List of timestamps
        date_row (list(datetime)) : List of dates
    """
    timestamp_row = []
    date_row = []

    timestamp = 1672531200 # unix timestamp for 01/01/2023

    for i in range(n):
        timestamp_row.append(timestamp)
        date_row.append(datetime.fromtimestamp(timestamp))
        timestamp += np.random.normal(1800, 1.0)
        timestamp = int(timestamp)
    return timestamp_row, date_row


def main():
    """df = pd.read_csv('../data/discord.csv')

    df = df.sort_values(by='timestamp')
    df.to_csv('../data/test.csv')

    # Get average time between messages
    total_messages = len(df)
    total_diff=0
    for i in range(1, total_messages):
        total_diff += (df['timestamp'].iloc[i] - df['timestamp'].iloc[i-1])

    ave_diff=total_diff / (total_messages - 1)

    print(ave_diff, 'seconds')
    print((ave_diff / 3600), 'hours')"""

    new_df = pd.read_csv('../data/cleaned.csv')


    print(new_df.groupby(['thread_id']).count())
    threads = new_df['thread_id'].unique().tolist()
    #print(threads)
    interleave_rate= 0.15
    for i in range(1, len(threads)):
        if i == 1:
            new_thread = interleave_helper(new_df.loc[new_df['thread_id'] == threads[0], ['id', 'thread_id']], new_df.loc[new_df['thread_id'] == threads[1], ['id', 'thread_id']], interleave_rate)
        
        else:
            new_thread = interleave_helper(new_thread, new_df.loc[new_df['thread_id'] == threads[i], ['id', 'thread_id']], interleave_rate)
            
        interleave_rate *= 0.9

    #print(new_thread)
    new_list = new_thread['thread_id'].tolist()

    mapping = createLabelMapping(threads)
    new_list = mapLabels(new_list, mapping)

    plt.scatter(range(len(new_list)), new_list, marker='x', color='blue')
    plt.xlabel('Message Index')
    plt.ylabel('Thread Number')
    plt.title('Message Distribution Across Threads in Synthesised Dataset')
    # plt.show()
    plt.savefig('../figures/synthesised_dist.png')
    #new_thread.to_csv('../data/test2.csv')

    sorted_df = new_df.set_index('id').loc[new_thread['id']].reset_index()
    timestamp_row, date_row = new_synthesiseDates(len(sorted_df))
    sorted_df['synthetic_dates'] = date_row
    sorted_df['timestamps'] = timestamp_row
    sorted_df['offsets'] = getTimestampOffsets(sorted_df['timestamps'])


    sorted_df['synthesised_day_of_week_str'], sorted_df['synthesised_day_of_week_int'], sorted_df['synthesised_days'], sorted_df['synthesised_months'], sorted_df['synthesised_years'] = splitDate(sorted_df['synthetic_dates'])
    sorted_df['synthesised_hours'] = splitTime(sorted_df['synthetic_dates'])
        

    sorted_df.to_csv('../data/cleaned_and_synthesised.csv', index=False)

if __name__ == "__main__":
    main()