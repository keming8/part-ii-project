import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_barplot(name, labels, data):
    """
    Plot a 4-way barplot for accuracy, precision, recall, f1 score

    Parameters:

        name (str) : Name of the plot
        labels (list) : Labels for X-axis
        data (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    data_a, data_p, data_r, data_f = data
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes[0,0].bar(labels, data_a, color = 'red')
    axes[0,0].set_title('Accuracy')

    axes[0,1].bar(labels, data_p, color = 'blue')
    axes[0,1].set_title('Precision')

    axes[1,0].bar(labels, data_r, color = 'green')
    axes[1,0].set_title('Recall')

    axes[1,1].bar(labels, data_f, color = 'yellow')
    axes[1,1].set_title('F1')
    
    plt.tight_layout()

    plt.savefig('../figures/' + name + '_barplot.png')
    #plt.show()
    plt.close()

def plot_heatmap(name, labels, data):
    """
    Plot a heatmap for the confusion matrix

    Parameters:

        name (str) : Name of the plot
        labels (list) : Labels for X-axis
        data (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels = labels, yticklabels = labels)
    plt.title(name + ' ' + 'Confusion Matrix Heatmap')
    plt.xlabel('Predicted Match')
    plt.ylabel('Actual Match')
    plt.tight_layout()
    plt.savefig('../figures/' + name + '_heatmap.png')
    #plt.show()
    plt.close()

# Plot to show whether there are class imbalances
def plot_class_barplot(name, df):
    """
    Plot a barplot to show the class imbalances

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    try:
        cnt = df['thread_id'].value_counts(sort=False).reset_index()
    except KeyError:
        cnt = df.value_counts(sort=False).reset_index()
    counts = cnt['count'].tolist()
    labels = cnt['thread_id'].tolist()
    labels = [str(label) for label in labels]
    plt.bar(labels, counts, color = 'lightgreen')
    plt.xlabel('Thread IDs')
    plt.xticks([])
    #plt.xticks(rotation = 90)
    plt.ylabel('Number of Messages')
    plt.title('Thread ID Distribution')
    plt.tight_layout()
    plt.savefig('../figures/' + name + '_class_balance' + '_barplot.png')
    plt.close()

def plot_user_barplot(name, df):
    """
    Plot a single barplot

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    users_per_thread = df.groupby('thread_id')['creator_id'].nunique()
    counts = users_per_thread.tolist()
    labels = np.arange(len(counts))
    plt.bar(labels, counts, color = 'lightgreen')
    plt.xlabel('Thread IDs')
    plt.xticks([])
    #plt.xticks(rotation = 90)
    plt.ylabel('Number of Creators')
    plt.title('Creators Per Thread')
    plt.tight_layout()
    plt.savefig('../figures/' + name + '_user' + '_barplot.png')
    plt.close()

def plot_thread_per_user_barplot(name, df):
    """
    Plot a single barplot

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    users_per_thread = df.groupby('creator_id')['thread_id'].nunique()
    counts = users_per_thread.tolist()
    labels = np.arange(len(counts))
    plt.bar(labels, counts, color = 'lightgreen')
    plt.xlabel('Thread IDs')
    plt.xticks([])
    #plt.xticks(rotation = 90)
    plt.ylabel('Number of Creators')
    plt.title('Users Per Thread')
    plt.tight_layout()
    plt.savefig('../figures/' + name + '_thread_per_user' + '_barplot.png')
    plt.close()

def plot_class_histogram(name, df):
    """
    Plot a histogram

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    try:
        cnt = df['thread_id'].value_counts(sort=False).reset_index()
    except KeyError:
        cnt = df.value_counts(sort=False).reset_index()
    print(cnt)
    counts = cnt['count'].tolist()
    print(counts)
    labels = cnt['thread_id'].tolist()
    labels = [str(label) for label in labels]

    plt.hist(counts, bins=len(labels), color='skyblue', edgecolor='black')
    #plt.bar(labels, counts, color = 'blue')
    #plt.xticks([])
    plt.xlabel('Number of Messages')
    #plt.xticks(rotation = 90)
    plt.ylabel('Number of Threads')
    plt.title('Thread ID Distribution')
    #plt.tight_layout()
    plt.savefig('../figures/' + name + '_class_balance' + '_histogram.png')
    plt.close()

def plot_user_histogram(name, df):
    """
    Plot a single histogram

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    users_per_thread = df.groupby('thread_id')['creator_id'].nunique()
    counts = users_per_thread.tolist()
    labels = np.arange(len(counts))
    

    plt.hist(counts, bins=len(labels), color='skyblue', edgecolor='black')
    #plt.bar(labels, counts, color = 'blue')
    #plt.xticks([])
    plt.xlabel('Number of Creators')
    #plt.xticks(rotation = 90)
    plt.ylabel('Number of Threads')
    plt.title('Thread ID Distribution')
    #plt.tight_layout()
    plt.savefig('../figures/' + name + '_user' + '_histogram.png')
    plt.close()

def plot_thread_per_user_histogram(name, df):
    """
    Plot a single histogram

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    # data is the 'thread_id' column
    users_per_thread = df.groupby('creator_id')['thread_id'].nunique()
    counts = users_per_thread.tolist()
    
    plt.hist(counts, bins=len(counts), color='skyblue', edgecolor='skyblue')
    plt.yscale('log')
    #plt.bar(labels, counts, color = 'blue')
    #plt.xticks([])
    plt.xlabel('Number of Threads')
    #plt.xticks(rotation = 90)
    plt.ylabel('Log Number of Creators')
    plt.title('Thread ID Distribution')
    #plt.tight_layout()
    plt.savefig('../figures/' + name + '_thread_per_user' + '_histogram.png')
    plt.close()

def plot_linegraph(name, x_val=None, y_val=None):
    """
    Plots a simple line graph

    Parameters:

        name (str) : Name of the plot
        x_val (list) : X values to plot
        y_val (list) : y values to plot

    Returns:

        None
    """
    x_val = [2, 4, 5, 10, 20, 50, 100]
    y_val = [0.0942622950819672, 0.16393442622950818, 0.15163934426229508, 0.21721311475409835, 0.2540983606557377, 0.4180327868852459, 0.4918032786885246]

    plt.plot(x_val, y_val, label = name)

    plt.xlabel('Percentage of Training Data')
    #plt.xticks(x_val)
    plt.ylabel('Accuracy')

    plt.show()
    #plt.savefig('../figures/' + name + '.png')

def plot_linegraph2(name, df):
    """
    Plot a linegraph with two separate lines

    Parameters:

        name (str) : Name of the plot
        df (pd.DataFrame) : Data to visualise

    Returns:

        None
    """
    x_val = [i for i in range(len(df))]
    y_val = df['train_loss']

    x2_val = [i for i in range(len(df))]
    y2_val = df['val_loss']

    plt.plot(x_val, y_val, label = 'Train Loss')
    plt.plot(x2_val, y2_val, label = 'Val Loss')

    plt.xlabel('Epochs')
    #plt.xticks(x_val)
    plt.ylabel('Loss')
    plt.xticks(range(0, len(df), 2))
    plt.legend()
    plt.show()
    # plt.savefig('../figures/' + name + '.png')

def show_chat(X_test, y_pred):
    """
    Meant to visualise the predicted chat to see if it is coherent
    Data should already be sorted in terms of message time
    This needs to be ensured as there are cases where the date/time of messages is not being taken into account when classifying

    Parameters:

        X_test (array-like) : Testing data to visualise
        y_pred (array-like) : Predicted labels for testing data

    Returns:

        None
    """

    pairs = zip(X_test['creator_id'], X_test['content'], y_pred)

    content_dict = {}

    for creator, content, thread in pairs:
        
        if thread in content_dict:
            content_dict[thread].append(str(creator) + ': ' + str(content) )
        else:
            content_dict[thread] = [str(creator) + ': ' + str(content)]

    df = pd.DataFrame.from_dict(content_dict, orient='index').transpose()

    df.to_csv('../data/visualise_chats_discord.csv')