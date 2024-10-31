import random
import ast
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import spacy
from bert_helpers import *


def pre_train_tests(X_train, X_test, y_train, y_test):
    """
    To identify bugs early on and correct them before training, preventing a wasted training job

    Parameters:

        X_train (array-like) : Training data
        X_test (array-like) : Testing data
        y_train (array-like) : Training labels
        y_test (array-like) : Testing labels

    Returns:

        None
    """

    # Check Data Shape
    assert len(X_train) == len(y_train), "Training data and labels have different sizes"
    assert len(X_test) == len(y_test), "Testing data and labels have different sizes"

    # Check for None types
    assert sum([1 if x is None else 0 for x in X_train]) == 0, 'Training data has None types'
    assert sum([1 if y is None else 0 for y in y_train]) == 0, 'Training labels have None types'
    assert sum([1 if x is None else 0 for x in X_train]) == 0, 'Testing data has None types'
    assert sum([1 if y is None else 0 for y in y_test]) == 0, 'Testing labels have None types'
    
    
    """# Check Label Leakage
    y_list = y_test.tolist()
    for X, y in zip(X_train, y_train):
        print(X_train.iloc[int(X)])
        assert not((X_test == X_train.iloc[(int(X))]).all(axis = 1).any()), "Label Leakage"""
        

def augmentX_SVM(X_train, vectoriser, augmentation='synonym', aug_prob=0.1):
    """
    Augment the training data to see what effects it has on the results

    Parameters:

        X_train (array-like) : Training data
        vectoriser : The vectoriser used for classification
        augmentation (str) : The type of augmentation to do on the training data
        aug_prob (float) : Probability of augmentation

    Returns:

        X_augmented (pd.Series) : Augmented training data
    """
    # Perturbations, same as post-train tests

    prob = aug_prob
    
    if augmentation == 'synonym':
        syn_aug = naw.SynonymAug(aug_p = prob)
        X_augmented = pd.Series([syn_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])
    
    elif augmentation == 'word_deletion':
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        X_augmented = pd.Series([del_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])
    
    elif augmentation == 'swap':
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        X_augmented = pd.Series([swap_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])
    
    elif augmentation == 'spelling':
        spell_aug = naw.SpellingAug(aug_p=prob)
        X_augmented = pd.Series([spell_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])

    elif augmentation == 'char_replacement':
        char_aug = nac.RandomCharAug(aug_char_p=prob)
        X_augmented = pd.Series([char_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])

    elif augmentation == 'keyboard':
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        X_augmented = pd.Series([keyboard_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])

    elif augmentation == 'antonym':
        ant_aug = naw.AntonymAug(aug_p = prob)
        X_augmented = pd.Series([ant_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])

    elif augmentation == 'split':
        split_aug = naw.SplitAug(aug_p=prob)
        X_augmented = pd.Series([split_aug.augment(' '.join(ast.literal_eval(text))) for text in X_train])
    
    X_augmented = X_augmented.apply(lambda x : ' '.join(x))
    X_augmented = vectoriser.transform(X_augmented)
    #print(X_randomCharacterReplacement)

    return X_augmented

def augmentX_BERT(X_train, augmentation='synonym', aug_prob=0.1):
    """
    Augment the training data to see what effects it has on the results

    Parameters:

        X_train (array-like) : Training data
        augmentation (str) : The type of augmentation to do on the training data
        aug_prob (float) : Probability of augmentation

    Returns:

        X_augmented (pd.Series) : Augmented training data
    """

    #print(content)
    prob = aug_prob
    
    if augmentation == 'synonym':
        syn_aug = naw.SynonymAug(aug_p = prob)
        X_augmented = [syn_aug.augment(text) for text in X_train]
    
    elif augmentation == 'word_deletion':
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        X_augmented = [del_aug.augment(text) for text in X_train]
    
    elif augmentation == 'swap':
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        X_augmented = [swap_aug.augment(text) for text in X_train]
    
    elif augmentation == 'spelling':
        spell_aug = naw.SpellingAug(aug_p=prob)
        X_augmented = [spell_aug.augment(text) for text in X_train]

    elif augmentation == 'char_replacement':
        char_aug = nac.RandomCharAug(aug_char_p=prob)
        X_augmented = [char_aug.augment(text) for text in X_train]

    elif augmentation == 'keyboard':
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        X_augmented = [keyboard_aug.augment(text) for text in X_train]

    elif augmentation == 'antonym':
        ant_aug = naw.AntonymAug(aug_p = prob)
        X_augmented = [ant_aug.augment(text) for text in X_train]

    elif augmentation == 'split':
        split_aug = naw.SplitAug(aug_p=prob)
        X_augmented = [split_aug.augment(text) for text in X_train]
    
    X_augmented = [x for l in X_augmented for x in l]
    return X_augmented

def augmentX_SVM_phase2(X_train, augmentation='synonym', aug_prob=0.1):
    """
    Augment the training data to see what effects it has on the results

    Parameters:

        X_train (array-like) : Training data
        augmentation (str) : The type of augmentation to do on the training data
        aug_prob (float) : Probability of augmentation

    Returns:

        X_augmented (pd.Series) : Augmented training data
    """

    #print(content)
    prob = aug_prob
    
    if augmentation == 'synonym':
        syn_aug = naw.SynonymAug(aug_p = prob)
        X_augmented = pd.Series([syn_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'word_deletion':
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        X_augmented = pd.Series([del_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'swap':
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        X_augmented = pd.Series([swap_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'spelling':
        spell_aug = naw.SpellingAug(aug_p=prob)
        X_augmented = pd.Series([spell_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'char_replacement':
        char_aug = nac.RandomCharAug(aug_char_p=prob)
        X_augmented = pd.Series([char_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'keyboard':
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        X_augmented = pd.Series([keyboard_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'antonym':
        ant_aug = naw.AntonymAug(aug_p = prob)
        X_augmented = pd.Series([ant_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'split':
        split_aug = naw.SplitAug(aug_p=prob)
        X_augmented = pd.Series([split_aug.augment(' '.join(text)) for text in X_train])
    
    X_augmented = X_augmented.apply(lambda x : ' '.join(x))
    nlp = spacy.load('en_core_web_sm')
    tokens = X_augmented.astype('unicode').apply(lambda text: tokenise_helper(nlp, text))
    #print(X_randomCharacterReplacement)
    tokens = tokens.to_list()
    return tokens

def augmentX_BERT_phase2(X_train, augmentation='synonym', aug_prob=0.1):
    """
    Augment the training data to see what effects it has on the results

    Parameters:

        X_train (array-like) : Training data
        augmentation (str) : The type of augmentation to do on the training data
        aug_prob (float) : Probability of augmentation

    Returns:

        X_augmented (pd.Series) : Augmented training data
    """

    #print(content)
    prob = aug_prob
    
    if augmentation == 'synonym':
        syn_aug = naw.SynonymAug(aug_p = prob)
        X_augmented = pd.Series([syn_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'word_deletion':
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        X_augmented = pd.Series([del_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'swap':
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        X_augmented = pd.Series([swap_aug.augment(' '.join(text)) for text in X_train])
    
    elif augmentation == 'spelling':
        spell_aug = naw.SpellingAug(aug_p=prob)
        X_augmented = pd.Series([spell_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'char_replacement':
        char_aug = nac.RandomCharAug(aug_char_p=prob)
        X_augmented = pd.Series([char_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'keyboard':
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        X_augmented = pd.Series([keyboard_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'antonym':
        ant_aug = naw.AntonymAug(aug_p = prob)
        X_augmented = pd.Series([ant_aug.augment(' '.join(text)) for text in X_train])

    elif augmentation == 'split':
        split_aug = naw.SplitAug(aug_p=prob)
        X_augmented = pd.Series([split_aug.augment(' '.join(text)) for text in X_train])
    
    X_augmented = X_augmented.to_list()
    return X_augmented

def tokenise_helper(nlp, text):
    """
    Helper function for tokenisation

    Parameters:

        nlp (spacy.Tokenizer) : The tokeniser to use
        text (str) : Text to tokenise

    Returns:

        tokenised (list) : List of tokens
    """

    docs = nlp(text)
    tokenised = [token.text for token in docs]
    return tokenised

def augmentY_SVM(y_train):
    """
    Augments labels for training to check for robustness

    Parameters:

        y_train (array-like) : Training labels

    Returns:

        y (list) : List of augmented y values
    """
        
    y_list = y_train.unique().tolist()
    y = randomAssignment(y_train, y_list)

    return y


def randomAssignment(y, y_labels, prob = 0.1):
    """
    Randomly assign a datapoint to a different class

    Parameters:

        y (array-like) : Array of y values
        y_labels (list) : List of all y values
        prob (float) : Probability of assigning randomly

    Return:

        new_y (list) : List of augmented y values

    """
    y_list = y.tolist()
    new_y = []
    for i in range(len(y_list)):
       
        if random.random() < prob:
            new_y.append(random.choice(y_labels))
        else:
            new_y.append(y_list[i])
    
    return new_y

def pairSentencesTest(X, y=None, bert=False):
    """
        Pairs sentences up for testing.

        Parameters:

            X (array-like) : Train features to pair.
            y (Union[None, array-like]) : Train labels.
            bert (bool) : Indicates if model is a BERT model.

        Returns:

            tuple:
                - token_pairs (list(str)) : The pairs of tokens obtained.
                - other_pairs (list(int)) : The pairs of CIDs and timestamps obtained.
                - labels (list(int)) : List of labels indicating a match or not.
        
        """
    # Pair up the sentences for classification

    feature_cols = list(X.columns)
    for i in ['id', 'tokens', 'content', 'refs']:
        if i in feature_cols:
            feature_cols.remove(i)

    token_pairs = []
    other_pairs = []
    # 1 if same class, 0 if not
    labels = []
    
    if bert:
        X_tokens = X['content'].to_list()
    else:
        X_tokens = X['tokens'].to_list()
    
    X_others = X[feature_cols].copy()
    
    # train_label_list = y_train.tolist() 
    train_label_list = y['thread_id'].tolist()
    for i in range(len(X)):
        for j in range(len(y)):
            if i != j:
                token_pairs.append((X_tokens[i], X_tokens[j]))
                other_pairs.append((list(X_others.iloc[i].values), list(X_others.iloc[j].values)))
                
                # Consider adding the reverse here
                if train_label_list[i] == train_label_list[j]:
                    labels.append(1)
                else:
                    labels.append(0)
    return token_pairs, other_pairs, labels
