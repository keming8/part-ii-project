import ast
from itertools import combinations
from sklearn.metrics import silhouette_score, pairwise_distances

def group(df, col):
    """
    Groups the dataframe by a specified column.

    Parameters: 

        df (DatFrame) : DataFrame to group.
        col (str) : Column to group by.

    Return:

        group_dict (dict) : Grouped dictionary
    """
    df = df[[col, 'tokens']]
    grouped = df.groupby(col)
    group_dict = {}

    for thread, texts in grouped:
        if thread in group_dict.keys():

            group_dict[thread] = group_dict['thread'].append(ast.literal_eval(texts['tokens'].tolist()[0]))
        
        else:
            group_dict[thread] = [ast.literal_eval(texts['tokens'].tolist()[0])]

    return group_dict

def group_bert(df, col):
    """
    Groups the dataframe by a specified column.

    Parameters: 

        df (DatFrame) : DataFrame to group.
        col (str) : Column to group by.

    Return:

        group_dict (dict) : Grouped dictionary
    """
    df = df[[col, 'content']]
    grouped = df.groupby(col)
    group_dict = {}

    for thread, texts in grouped:
        if thread in group_dict.keys():

            group_dict[thread] = group_dict['thread'].append(texts['content'].tolist()[0])
        
        else:
            group_dict[thread] = [texts['content'].tolist()[0]]
    
    return group_dict

def get_jaccard(t1, t2):
    """
    Gets the Jaccard score for two lists of lists

    Parameters:

        t1 (list(list)) : First list
        t2 (list(list)) : Second list

    Returns:

        intersect_len / union_len (float) : Jaccard score for the two inputs.
    """
    # Convert list of lists to set of lists
    t1 = set(t1)
    t2 = set(t2)

    intersect_len = len(t1.intersection(t2))
    union_len = len(t1.union(t2))

    if union_len == 0:
        return 0.0
    else:
        return intersect_len / union_len
    
def compute_jaccard(groups):
    """
    Computes the Jaccard score for each pair within the group

    Parameters:

        groups (dict) : Groups to get Jaccard scores of.

    Return:

        jaccard_dict (dict) : Dictionary of Jaccard scores.
    """
    # For each thread label
    labels = list(groups.keys())
    jaccard_dict = dict()
    for label1, label2 in combinations(labels, 2):
        # For each list in thread label
        jaccard_scores = []
        
        for list1 in groups[label1]:
            for list2 in groups[label2]:
                jaccard = get_jaccard(list1, list2)
                jaccard_scores.append(jaccard)
        jaccard_score = sum(jaccard_scores) / len(jaccard_scores)
        if jaccard_score > 0.6:
            if label1 in jaccard_dict.keys():
                jaccard_dict[label1].add(label2)
            else:
                jaccard_dict[label1] = {label2}
            """print(label1)
            print(label2)
            print('Jaccard Score: ', jaccard_score)
            print('----------')"""
    return jaccard_dict

def compute_jaccard_vectorised(groups, vectorised, tokens):
    """
    Computes the Jaccard score using vectorised tokens for each pair within the group

    Parameters:

        groups (dict) : Groups to get Jaccard scores of.
        vectorised (scipy.sparse) : Sparse matrix of vectorised values.
        tokens (list) : List of all tokens

    Return:

        jaccard_dict (dict) : Dictionary of Jaccard scores.
    """
    # For each thread label
    labels = list(groups.keys())
    jaccard_dict = dict()
    for label1, label2 in combinations(labels, 2):
        # For each list in thread label
        jaccard_scores = []
        
        for list1 in groups[label1]:
            for list2 in groups[label2]:
 
                c1_index = tokens.index(list1)
                c2_index = tokens.index(list2)
                c1 = vectorised.getrow(c1_index)
                c2 = vectorised.getrow(c2_index)
                
                # Convert sparse matrices to sets of non-zero indices
                non_zero_indices1 = zip(*c1.nonzero())
                non_zero_indices2 = zip(*c2.nonzero())
              

                jaccard = get_jaccard(non_zero_indices1, non_zero_indices2)
                jaccard_scores.append(jaccard)

        js = sum(jaccard_scores) / len(jaccard_scores)
        #print(js)
        if js > 0.22:
            if label1 in jaccard_dict.keys():
                jaccard_dict[label1].add(label2)
            else:
                jaccard_dict[label1] = {label2}
            """print(label1)
            print(label2)
            print('Jaccard Score: ', js)
            print('----------')"""
        
    return jaccard_dict

def merge_dict(jaccard_dict):
    """
    Merge common values between dictionary keys

    Parameters:

        jaccard_dict (dict) : Dictionary of Jaccard scores

    Return:

        jaccard_dict (dict) : Dictionary of Jaccard scores.
    """

    new_dict = jaccard_dict.copy()
    # Check for values being keys
    for i in jaccard_dict.keys():
        for j in jaccard_dict[i]:
            if j in jaccard_dict.keys():
                new_dict[i] = jaccard_dict[i].union(jaccard_dict[j])
                if j in new_dict.keys():
                    del new_dict[j]

    # Check for similar values
    for i in jaccard_dict.keys():
        for j in jaccard_dict.keys():
            if i != j and i in new_dict.keys() and j in new_dict.keys():
                common_values = new_dict[i].intersection(new_dict[j])
                if len(common_values) > 0:
                    new_dict[i] = new_dict[i].union(new_dict[j])
                    del new_dict[j]
    
    return new_dict

def flip_dict(dict):
    """
    Flip keys and values in dictionary

    Parameters:

        dict (dict) : Dictionary to flip

    Returns:

        new_dict (dict) : Dictionary with keys and values flipped.
    """
    new_dict = {value : key for key, values in dict.items() for value in values}
    return new_dict
    
def merge_threads(df, dict):
    """
    Merge similar threads

    Parameters:

        df (pd.DataFrame) : The dataset.
        dict (dict) : Dictionary of Jaccard scores

    Returns:

        df (pd.DataFrame) : Updated DataFrame
    """
    # Following the jaccard dictionary, merge the threads by assigning diff threads based on it
    
    for i in range(len(df)):
        
        if df.loc[i, 'Predicted'] in dict.keys():
            df.loc[i, 'Predicted'] = dict[df.loc[i, 'Predicted']]
    return df

def eval_df(df, tokens, vectorised):
    """
    Evaluates the given dataset by using Jaccard scores

    Parameters:

        df (pd.DataFrame) : The dataset.
        tokens (list) : List of all tokens.
        vectorised (scipy.sparse) : Sparse matrix of vectorised tokens.

    Returns:

        jaccard_eval_dict (dict) : Dictionary of threads with Jaccard scores
    """
    predicted_groups = group(df, 'Predicted')
    actual_groups = group(df, 'Actual')
    jaccard_eval_dict = dict()
    # Compute Jaccard between the predicted and actual groups
    for id1, group1 in predicted_groups.items():
        for id2, group2 in actual_groups.items():
            # Need to take the average of the Jaccard scores of the 2 groups
            jaccard_scores = []
            for g1 in group1:
                for g2 in group2:
                    c1_index = tokens.index(g1)
                    c2_index = tokens.index(g2)
                    c1 = vectorised.getrow(c1_index)
                    c2 = vectorised.getrow(c2_index)
                    
                    # Convert sparse matrices to sets of non-zero indices
                    non_zero_indices1 = zip(*c1.nonzero())
                    non_zero_indices2 = zip(*c2.nonzero())

                    jaccard = get_jaccard(non_zero_indices1, non_zero_indices2)
                    jaccard_scores.append(jaccard)
            jaccard_score = sum(jaccard_scores) / len(jaccard_scores)
            jaccard_eval_dict[(id1, id2)] = jaccard_score
    
    return jaccard_eval_dict

def eval_pairs(df):
    """
    Evaluates pairs in the dataset

    Parameters:

        df (pd.DataFrame) : The dataset to evaluate

    Returns:

        new_pred (list) : List of predicted classifications
        new_actual (list) : List of actual match / mistmatch
    """
    predicted = df['Predicted'].tolist()
    actual = df['Actual'].tolist()

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    total = 0

    assert len(predicted) == len(actual), 'Length of predictions and actual must be the same'
    new_pred = []
    new_actual = []
    for i in range(len(predicted)):
        for j in range(len(predicted)):
            # True positive
            if predicted[i] == predicted[j] and actual[i] == actual[j]:
                true_pos += 1
            # True negative
            elif predicted[i] != predicted[j] and actual[i] != actual[j]:
                true_neg += 1
            # False positive
            elif predicted[i] == predicted[j] and actual[i] != actual[j]:
                false_pos += 1
            # False negative
            elif predicted[i] != predicted[j] and actual[i] == actual[j]: 
                false_neg += 1
            total += 1
            new_pred.append(predicted[i] == predicted[j])
            new_actual.append(actual[i] == actual[j])
    
    accuracy = (true_pos + true_neg) / total
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)

    return new_pred, new_actual

def predict_eval(df, X_test_vectorised, X_test_tokens):
    """
    Evaluates the dataset by first merging threads, then doing pairwise evaluation.

    Parameters:

        df (pd.DataFrame) : The dataset to evaluate
        X_test_vectorised (scipy.sparse) : Sparse matrix of vectorised tokens
        X_test_tokens : List of tokens

    Returns:

        None
    """
    groups = group(df, 'Predicted')
    #jaccard_dict = compute_jaccard_vectorised(groups, X_test_vectorised, X_test_tokens)
    jaccard_dict = compute_jaccard(groups)
    jaccard_dict = merge_dict(jaccard_dict)
    flipped_jaccard_dict = flip_dict(jaccard_dict)

    print(len(df['Predicted'].unique()))
    df = merge_threads(df, flipped_jaccard_dict)
    print(len(df['Predicted'].unique()))
    df['Predicted'] = df['Predicted'] // 56
    print(len(df['Predicted'].unique()))
    df.to_csv('../data/test_seen_merged.csv')
    
    jaccard_eval_dict = eval_df(df, X_test_tokens, X_test_vectorised)
    #print({k: v for k, v in sorted(jaccard_eval_dict.items(), key=lambda item: item[1])})
    eval_pairs(df)

def predict_eval_bert(df):
    """
    Evaluates the dataset by first merging threads, then doing pairwise evaluation.

    Parameters:

        df (pd.DataFrame) : The dataset to evaluate

    Returns:

        None
    """
    groups = group_bert(df, 'Predicted')
    jaccard_dict = compute_jaccard(groups)
    jaccard_dict = merge_dict(jaccard_dict)
    flipped_jaccard_dict = flip_dict(jaccard_dict)
    print(len(df['Predicted'].unique()))
    #df = merge_threads(df, flipped_jaccard_dict)
    print(len(df['Predicted'].unique()))
    df['Predicted'] = df['Predicted'] // 15
    print(len(df['Predicted'].unique()))
    df.to_csv('../data/test_seen_merged.csv')
    #jaccard_eval_dict = eval_df(df, X_test_tokens, X_test_vectorised)
    #print({k: v for k, v in sorted(jaccard_eval_dict.items(), key=lambda item: item[1])})
    eval_pairs(df)

def get_silhouette(df, vectoriser):
    """
    Calculates silhouette score for threads in the dataset

    Parameters:

        df (pd.DataFrame) : The dataset to evaluate
        vectoriser : Vectoriser used for model

    Returns:

        None
    """
    data = df['tokens'].tolist()
    data_vectorised = vectoriser.transform(data)
    labels = df['thread_id'].tolist()
    distances = pairwise_distances(data_vectorised)
    silhouette_avg = silhouette_score(distances, labels)

    print(silhouette_avg)