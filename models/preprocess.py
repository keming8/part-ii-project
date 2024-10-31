import pandas as pd
import re
import numpy as np
import emoji
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from string import punctuation
from nltk.corpus import stopwords
from datetime import datetime
from collections import Counter
from spellchecker import SpellChecker


def is_numeric(id):
    """
    Checks if value is numeric

    Parameters:
        id : Value to be checked

    Returns:
        True if value can be converted to a float, else False
    """
    try:
        float(id)
        return True
    except ValueError:
        return False
    
def extract_pids(text):
    """
    Extracts Post IDs from links in text.

    Parameters:

        text (str) : Text to preprocess and extract PID from 
    
    Returns:

        refs (set) : Set of references
    """
    refs = set()
    text = text.lower()
    text = text.split()

    for w in text:
        if 'pid=' in w:
            w = w.split('pid=')[1]
            w = w.split('#')[0]
            # Remove any potential quotation marks
            w = w.strip("'")
            w = w.strip('"')
            if is_numeric(w):
                refs.add(w)
    return refs

def clean_text(text):
    """
    Cleans text by removing LINK tags, punctuation etc

    Parameters:

        text (str) : Text to preprocess

    Returns:

        text (str) : Preprocessed text
    """
    #text.replace('***IMG***', '').replace('***CITING***', '')
    text = text.replace('IMG', '')
    text = text.replace('CITING', '')
    text = text.replace('LINK', '') # Remove img, link and citing tags
    # Convert to lower case
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text) # Remove links
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation

    text = ' '.join(text.split())  # Remove extra spaces, tabs, and new lines
    return text

# Removes stopwords
def remove_stopwords(text):
    """
    Removes stopwords from text

    Parameters:

        text (str) : Text to remove stop words from

    Returns:

        noStopwords (str) : Text without any stop words
    """
    noStopwords = []
    text = text.split()
    for w in text:
        if w not in stopwords.words('english'):
               noStopwords.append(w)
    # Put back into a sentence to be used by vectorizer
    noStopwords = ' '.join(noStopwords)
    return noStopwords


# Remove non alphanumeric characters
def remove_nonalphanumeric(text):
    """
    Remove non-alphanumeric characters from text

    Parameters:

        text (str) : Text to remove non-alphanumeric characters

    Returns:

        noNonAlpha (text) : Text without any alphanumeric characters
    """
    noNonAlpha = []
    text = text.split()
    for w in text:
         if w.isalnum():
              noNonAlpha.append(w)
    noNonAlpha = ' '.join(noNonAlpha)
    return noNonAlpha

def binariseCids(cid_col, refs_col):
    """
    Puts CIDs into one-hot encoded form

    Parameters:

        cid_col (pd.Series) : Column containing CID
        refs_col (pd.Series) : Column containing CIDs of messages replied to by each message

    Returns:

        label_matrix.tolist() (list) : List of one hot encoded CIDs

    """
    refs_col = [{int(x) for x in s} for s in refs_col]

    new_col = []
    for cid, refSet in zip(cid_col, refs_col):
        refSet.add((cid))
        new_col.append(refSet)
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(new_col)
    
    return label_matrix.tolist()

def preprocessCid(cid_col, refs_col):
    """
    Gets longest length of CID list for padding

    Parameters:

        cid_col (pd.Series) : Column containing CID
        refs_col (pd.Series) : Column containing CIDs of messages replied to by each message

    Returns:

        new_col (pd.Series) : Column of reference sets
        maxLen (int) : Length of longest list

    """
    new_col = []
    maxLen = 1
    for cid, refSet in zip(cid_col, refs_col):
        refSet.add(cid)
        new_col.append(list(refSet))
        if len(list(refSet)) > maxLen:
            maxLen = len(list(refSet))
    return new_col, maxLen

def padCids(cid_col, maxLen):
    """
    Pads padded list to the longest length

    Parameters:

        cid_col (pd.Series) : Column containing CID
        maxLen (int) : Value to pad to

    Returns:

        cid_col (list) : List of padded lists, padded to length of maxLen

    """
    for l in cid_col:
        while len(l) < maxLen:
            l.append('0')
    return cid_col

def splitDate(date_col):
    """
    Gets day of week, days, months, years from data

    Parameters:

        date_col (pd.Series) :  Column of date values for each message

    Returns:
    
        dayOfWeek_str (list) : List of string representations of day of week
        dayOfWeek_int (list) : List of int representations of day of week
        day (list) : List of days
        months (list) : List of months
        years (list) : List of years
    """
    dayOfWeek_str = []
    dayOfWeek_int = []

    days = []
    months = []
    years = []

    for i in date_col:
        # Convert to datetime object
        date = datetime.strptime(str(i).split()[0], '%Y-%m-%d')

        # Day of week from 0 to 6
        dayOfWeek = date.weekday()
        dayOfWeek_int.append(dayOfWeek)

        # Day of week as string
        dayOfWeek = date.strftime('%A')
        dayOfWeek_str.append(dayOfWeek)

        # Days
        day = date.strftime('%d')
        days.append(day)

        # Months
        month = date.strftime('%m')
        months.append(month)

        # Years
        year = date.strftime('%Y')
        years.append(year)

    return dayOfWeek_str, dayOfWeek_int, days, months, years

def splitTime(date_col):
    """
    Gets hours out of date column

    Parameters:

        date_col (pd.Series) : Column of date values for each message
    
    Returns:

        hours (list) : List of hours    
    """
    hours = []

    for i in date_col:
        time = str(i).split()[1]
        hour = time.split(':')[0]
        hours.append(int(hour))
    
    return hours

def getCreatorOfRef(data, refs_col, pid_col, cid_col):
    """
    Get CIDs of referenced messages

    Parameters:

        data (pd.DataFrame) : The dataset
        refs_col (str) : Name of column for references
        pid_col (str) : Name of column for PIDs
        cid_col (str) : Name of column for CIDs

    Returns:

        new_ref_cid_col (list) : New list containing CIDs of referenced messages
    """
    # refs_col consists of sets
    new_ref_cid_col = []
    
    pid_list = data[pid_col].tolist()
    
    for ref_set in data[refs_col]:
        ref_cids = set()
        for ref in ref_set:
            r = int(ref)
            if r in pid_list:
                # Find the row in dataset that has that PID
                row = data[data[pid_col] == r]
                
                for val in row[cid_col].values:
                    ref_cids.add(val)
                
        new_ref_cid_col.append(ref_cids)
    
    return new_ref_cid_col
    
def correctSpelling(spell, tokens):
    """
    Corrects spelling of text

    Parameters:

        spell (SpellChecker) : The SpellChecker object
        tokens (list) : List of tokens

    Returns:

        spellchecked (list) : List of tokens, with spellings fixed
    """
    spellchecked = [spell.correction(token) for token in tokens]
    return spellchecked

def get_pos_tag_counts(row):
    """
    Get counts for each POS tag
    
    Parameters:
        row (pd.Series) : Row in dataset

    Returns:
        pos_tag_counts (Counter) : Counter object of counts for each POS tag
    """
    pos_tag_counts = Counter(row)
    return pos_tag_counts

def checkifEmoji(col):
    """
    Checks if emojis exist in dataset

    Parameters:

        col (pd.Series) : Column of text to check

    Returns:

        emoji_list (list) : Binary list indicating whether emojis exist
    """
    emoji_list = []
    for i in col:
        if emoji.emoji_count(i) > 0:
            emoji_list.append([1])
        else:
            emoji_list.append([0])
    return emoji_list

def checkifEmoticon(col):
    """
    Checks if emoticons exist in dataset

    Parameters:

        col (pd.Series) : Column of text to check

    Returns:

        emoji_list (list) : Binary list indicating whether emoticons exist
    """
    emoji_list = []
    emoticon_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P)'
    for i in col:
        if re.search(emoticon_pattern, i):
            emoji_list.append([1])
        else:
            emoji_list.append([0])
    
    return emoji_list

import spacy
from tqdm import tqdm


def tokenise_helper(nlp, text):
    """
    Helper function for tokenisation, retrieves tokens and POS tags

    Parameters:
        
        nlp (spacy.Toknizer) : The tokeniser object
        text (str) : Text to be tokenised

    Returns:

        tokens (list) : List of tokens
        pos (list) : List of POS tags for each token
    """
    docs = nlp(text)
    tokens = [token.text for token in docs]
    pos = [token.pos_ for token in docs]
    return tokens, pos
    
def lemmatise_helper(nlp, text):
    """
    Helper function for lemmatisation, retrieves lemma and POS tags

    Parameters:
        
        nlp (spacy.Toknizer) : The tokeniser object
        text (str) : Text to be tokenised

    Returns:

        lemmas (list) : List of lemmas
        pos (list) : List of POS tags for each lemma
    """
    docs = nlp(text)
    lemmas = [token.lemma_ for token in docs]
    pos = [token.pos_ for token in docs]
    return lemmas, pos

def tokenise(df):
    """
    Tokenises, or lemmatises, text in the dataset

    Parameters:
        
        df (pd.DataFrame) : The dataset

    Returns:

        None
    """
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("emoji", first=True)
    tqdm.pandas()
    # Using tokens
    """df['tokens'] = list(tqdm(df['content'].astype('unicode').progress_apply(lambda x: [token.text for token in nlp(x)])))
    # Get POS tags
    df['postags'] = list(tqdm(df['content'].astype('unicode').progress_apply(lambda x: [token.pos_ for token in nlp(x)])))"""

    df['tokens'], df['postags'] = zip(*tqdm(df['tokens'].astype('unicode').progress_apply(lambda text: tokenise_helper(nlp, text))))
    """df['tokens'], df['postags'] = zip(*tqdm(df['tokens'].astype('unicode').progress_apply(lambda text: lemmatise_helper(nlp, text))))"""

def main():
    df = pd.read_csv('../data/crpytography.csv')
    # Drop rows with NA
    df = df.dropna()

    # Deal with just threads as labels and content
    thread_ids = df['thread_id']
    content = df['content']

    # Try focusing on just the content and thread_ids first
    # First need to clean the data

    refs = content.map(extract_pids)
    # Get referenced posts
    df['refs'] = refs
    # Follow referenced post to get creator id of that post
    df['ref_cids'] = getCreatorOfRef(df, 'refs', 'id', 'creator_id')
    """new_cid_col = []
    pid_list = df['id'].tolist()
    for i, row in df.iterrows():
        new_cid_col.append(getCreatorofRef(df, row, row['refs'], pid_list, 'id', 'refs', 'creator_id'))
    df['ref_cids'] = new_cid_col""" 
    df['binarised'] = binariseCids(df['creator_id'], df['ref_cids'])
    df['binarised_pids'] = binariseCids(df['id'], df['refs'])
    df['binarised_cid_single'] = binariseCids(df['creator_id'], [set() for _ in range(len(df))])
    df['all_cids'], maxLen = preprocessCid(df['creator_id'], df['ref_cids'])
    df['all_cids'] = padCids(df['all_cids'], maxLen)
    df['all_pids'], maxLen = preprocessCid(df['id'], df['refs'])
    df['all_pids'] = padCids(df['all_pids'], maxLen)
    #df['day_of_week_str'], df['day_of_week_int'], df['days'], df['months'], df['years'] = splitDate(df['created_on'])
    #df['hours'] = splitTime(df['created_on'])
    df['emojis'] = checkifEmoji(df['content'])
    df['emoticons'] = checkifEmoticon(df['content'])
    df['emojis_and_emoticons'] = [[x[0] | y[0]] for x, y in zip(df['emojis'], df['emoticons'])]

    content = content.map(clean_text)
    #content = content.map(remove_nonalpha)
    # Would this be necessary? Removes any emojis
    content_svm = content.map(remove_nonalphanumeric)
    content_svm = content.map(remove_stopwords)

    df['tokens'] = content_svm
    df['content'] = content


    # Sort into threads, then by dates
    sorted_df = df.sort_values(by = ['thread_id', 'created_on'], ascending = True)
    # sorted_df = df.sort_values(by = ['timestamp'], ascending = True)
    #sorted_df['created_on'] = pd.to_datetime(sorted_df['created_on'])
    #print(sorted_df.dtypes)

    # -- document length filtering, uncomment if required --
    # Removing outliers in terms of length of message
    """data = pd.read_csv('../data/hacking_tools.csv')

    X = data['content']

    # Compute document lengths
    document_lengths = np.array([len(document.split()) for document in X])

    # Calculate statistical measures
    mean_length = np.mean(document_lengths)
    std_dev_length = np.std(document_lengths)
    quantile_25 = np.percentile(document_lengths, 25)
    quantile_75 = np.percentile(document_lengths, 75)

    # Define thresholds based on statistical measures
    lower_threshold = quantile_25 - 1.5 * (quantile_75 - quantile_25)
    upper_threshold = quantile_75 + 1.5 * (quantile_75 - quantile_25)

    # Identify indices of documents outside the thresholds
    outlier_indices = np.where((document_lengths < lower_threshold) | (document_lengths > upper_threshold))[0]

    # Remove outliers from the dataset
    filtered_data = pd.DataFrame(data.iloc[i] for i in range(len(X)) if i not in outlier_indices)
    print(len(data))
    print(len(filtered_data))
    filtered_data.to_csv('../data/hacking_tools_filtered.csv')"""

    # Tokenise
    tokenise(sorted_df)

    # Count the POS tags
    sorted_df['pos_tag_counts'] = sorted_df['postags'].apply(get_pos_tag_counts)
    scaler = MinMaxScaler()
    
    # Convert counter values into pd.series
    pos_tags = (sorted_df['pos_tag_counts'].apply(pd.Series)).fillna(0)

    pos_tags_normalised = pd.DataFrame(scaler.fit_transform(pos_tags), columns=pos_tags.columns)

    sorted_df = pd.concat([sorted_df, pos_tags_normalised], axis=1)
    
    # -- spellchecking, uncomment if required --
    """ # Do spell-checking
    spell = SpellChecker()
    sorted_df['spellchecked'] = sorted_df['tokens'].apply(lambda tokens : correctSpelling(spell, tokens))
    """

     # -- word frequency filtering, uncomment if required --
    """all_words = []
    for row in sorted_df['tokens']:
        for word in row:
            all_words.append(word)
    
    word_freqs = Counter(all_words)

    # Choose a cutoff of 3
    filtered_words = [word for word, freq in word_freqs.items() if freq >= 3]
    
    new_content = []
    for row in sorted_df['tokens']:
        new_row = [word for word in row if word in filtered_words]
        new_content.append(new_row)

    sorted_df['tokens'] = new_content"""

    # Removing those with length = 0
    sorted_df = sorted_df[sorted_df['tokens'].apply(lambda x : len(str(x)) > 0)]

    sorted_df.replace('', np.nan, inplace=True)
    sorted_df = sorted_df.dropna()
    sorted_df.to_csv('../data/cleaned.csv')
    # sorted_df.to_csv('../data/discord_test_hacking2_cleaned.csv')

if __name__ == '__main__':
    main()