import ast
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.optim import Adam
from visualise import *
from testing_framework import *
from phase2_eval import *
from bert_helpers import *
from bert_models import *

class Phase2BERT():
    def __init__(self,
                 pairwise_dataset,
                 thread_dataset, 
                 model,
                 tokenizer,
                 feature_cols = ['id', 'content', 'creator_id', 'refs', 'timestamps'], 
                 batch_size=32,
                 learning_rate=0.001,
                 num_epochs=1,
                 stratify = True, 
                 few_shot = 1, 
                 augmentX = None, 
                 augmentY = False, 
                 aug_prob=0.1,
                 truncation_type='head-only',
                 model_name='custom',
                 classify='logits_median',
                 train_min = 4000,
                 val_min = 1000):
        """
        Initialise Phase 2 code. 

        Parameters:     

            pairwise_dataset (pd.DataFrame) : Dataset used for training and validation. Pairwise Classification.
            thread_dataset (pd.DataFrame) : Dataset used for testing. Thread Classification.
            learning_rate (float) : The size of the step taken during optimisation to update model parameters. Must be strictly positive.
            batch_size (int) : The size of each training batch. Must be strictly positive.
            num_epochs (int) : The number of epochs to train for. Must be strictly positive. 
            feature_cols (List[str]) : List of feature columns to be used for feature vectors.
            model_name (str) : The name of the BERT model to use. Valid options are 'bert', 'roberta', 'albert', 'modular' and 'custom'. 
            truncation_type (Union[None, str]) : The method of truncating sequences longer than 512 tokens. Valid options are None, 'head-only', 'head-and-tail' and 'tail-only'.
            stratify (bool) : States whether to do a stratified train-test-split or not.
            few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
            augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
            augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
            aug_prob (float) : Probability of pre-train augmentations. Must be strictly between 0.0 and 1.0.
            classify (str) : Classify method. Valid options are 'voting1', 'voting2', 'decision_function_max', 'decision_function_median' and 'decision_function_similarity_threshold'.
            train_min (int) : Minimum training samples for each class.
            val_min (int) : Minimum validation samples for each class.

        Returns: 

            None

        """
        
        self.pairwise_dataset = pairwise_dataset
        self.thread_dataset = thread_dataset
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.stratify = stratify
        self.few_shot = few_shot
        self.augmentX = augmentX
        self.augmentY = augmentY
        self.aug_prob = aug_prob
        self.truncation_type = truncation_type
        self.classify = classify
        self.train_min = train_min
        self.val_min = val_min


    def nearest_5_min(self, time_diff):
        """
        Rounds off the difference in timestamps to the nearest 5 minutes (300 seconds).

        Parameters:

            time_diff (float) : The difference in timestamps between messages.

        Returns:

            rounded (int) : The rounded value.
            
        """
        rounded = round(time_diff // 300) * 300
        return rounded

    def pairSentences(self, X_train, y_train=None, X_test=None, y_test=None, train=True):
        """
        Pairs sentences up for classification. 

        Parameters:

            X_train (array-like) : Train features to pair.
            y_train (Union[None, array-like]) : Train labels.
            X_test (Union[None, array-like]) : Test features to pair.
            y_test (Union[None, int]) : Test labels.
            train (bool) : Indicates if pairing for training or testing data.

        Returns:

            tuple:
                If training data is paired:
                - token_pairs (list(str)) : The pairs of tokens obtained.
                - other_pairs (list(int)) : The pairs of CIDs and timestamps obtained.
                - labels (list(int)) : List of labels indicating a match or not.
                If testing data is paired:
                - token_pairs (list(str)) : The pairs of tokens obtained.
                - other_pairs (list(int)) : The pairs of CIDs and timestamps obtained.
        
        """
        # Pair up the sentences for classification

        # X_train, y_train will be pandas series
        # X_test is a list of tokens
        # y_test is a single label
        
        feature_cols = list(X_train.columns)
        for i in ['id', 'content', 'refs']:
            if i in feature_cols:
                feature_cols.remove(i)
        
        token_pairs = []
        other_pairs = []
        # 1 if same class, 0 if not
        labels = []
        
        X_train_tokens = X_train['content']
        X_train_others = X_train[feature_cols].copy()
    
        # train_label_list = y_train.tolist()
        if train:
            train_label_list = y_train.tolist()
            for i in range(len(X_train)):
                for j in range(len(X_train)):
                    if i != j:
                        token_pairs.append((X_train_tokens.iloc[i], X_train_tokens.iloc[j]))
                        #token_pairs.append(X_train_tokens.iloc[i] + ' [SEP] ' + X_train_tokens.iloc[j])
                        
                        other_pairs.append((list(X_train_others.iloc[i].values), list(X_train_others.iloc[j].values)))

                        # Consider adding the reverse here
                        if train_label_list[i] == train_label_list[j]:
                            labels.append(1)
                        else:
                            labels.append(0)
            
            return token_pairs, other_pairs, labels
        
        elif y_train is not None and X_test is not None:
            feature_cols.remove('new_thread_id')
            X_test_tokens = X_test['content']
            X_test_others = X_test[feature_cols].copy()
            for i in range(len(X_train)):
                token_pairs.append((X_test_tokens, X_train_tokens.iloc[i]))
                #token_pairs.append(X_train_tokens.iloc[i] + ' [SEP] ' + X_train_tokens.iloc[j])
                other_pairs.append((list(X_test_others.tolist()), list(X_train_others.iloc[i].values)))

                
            return token_pairs, other_pairs
        

    def predict(self, test_seen, curr):
        """
        First stage of Phase 2. 
        Predict checks to see if a message is a reply to a previous message within test_seen.

        Parameters:

            test_seen (list(pd.Series)) : List of seen messages in the test set.
            curr (pd.Series) : Current message being analysed.

        Returns:

            thread (int) : The thread to classify curr as. Can be 0.
        
        """
        # If current post is a reply to a previous post
        refset = ast.literal_eval(curr['refs'])
        
        thread=None

        # If message is a reference
        if len(refset) > 1:
            ref = refset.difference({curr['id']})
            
            # Post may reference more than one previous post. Find one that's in the training set
            for r in ref:
                try:
                    # Get the thread that this post is classified to
                    thread = test_seen[test_seen['id'] == int(r)]['new_thread_id'].values[0]
                    #id = test_seen[test_seen['id'] == int(r)]['id'].values[0]

                    return thread
                except:
                    continue

        return thread
    
    def classify_voting1(self, all_preds, test_threads):
        """
        Classify Method, voting 1.
        Voting based on the use of predictions from classifier. Takes the thread with the largest count.

        Parameters:

            test_threads (list) : List of test threads.
            all_preds (list) : List of predictions obtained from model.

        Returns:

            thread (int) : Classified thread.
        """
        # Using the predictions and counting from there
        thread_preds = dict()

        assert len(all_preds) == len(test_threads), 'Preds and labels should have the same length'

        for i in range(len(test_threads)):
            # For new threads
            if all_preds[i] == 0:
                thread_preds[0] = thread_preds.get(0, 0) + 1
            # For existing threads
            else:
                thread_preds[test_threads[i]] = round(thread_preds.get(test_threads[i], 0) + 1 - (self.decay * 0.5))

        # Compare_1
        thread = max(thread_preds.items(), key=lambda item: (item[1], -item[0]))[0]

        return thread
    
    def classify_voting2(self, all_preds, test_threads):
        """
        Classify Method, voting 2.
        Checks to see if predicted 0s outweigh 1s, before making classifications.

        Parameters:

            test_threads (list) : List of test threads.
            all_preds (list) : List of predictions obtained from model.

        Returns:

            thread (int) : Classified thread.
        """
         # Using the predictions and counting from there
        thread_preds = dict()

        assert len(all_preds) == len(test_threads), 'Preds and labels should have the same length'

        for i in range(len(test_threads)):
            # For new threads
            if all_preds[i] == 0:
                thread_preds[0] = thread_preds.get(0, 0) + 1
            # For existing threads
            else:
                thread_preds[test_threads[i]] = round(thread_preds.get(test_threads[i], 0) + 1 - (self.decay * 0.5))

        # Compare_2
        # Check if it's to be allocated to a new thread
        if thread_preds.get(0, 0) > sum(thread_preds.values()) - thread_preds.get(0, 0):
            thread = 0
        else:
            if 0 in thread_preds.keys():
                del thread_preds[0]
            thread = max(thread_preds.items(), key=lambda item: (item[1], -item[0]))[0]
        
        return thread
    
    def classify_logits_max(self, all_logits, test_threads):
        """
        Classify Method, logits max. 
        Takes the absolute max value of each class, compares them. 

        Parameters:

            all_logits (torch.Tensor) : Logits from output of model.
            test_threads (list) : List of test threads.

        Returns:

            thread (int) : Classified thread.
        """
        zero_max = -float('inf')
        one_max = -float('inf')
        zero_max_ind = -1
        one_max_ind = -1
        logit = all_logits[0]
        for i, t in enumerate(logit):
            if t[0] > zero_max:
                zero_max = t[0]
                zero_max_ind = i
            if t[1] > one_max:
                one_max = t[1]
                one_max_ind = i
        
        zero_list = []
        one_list = []
        logit = all_logits[0]
        for t in logit:
            zero_list.append(t.numpy()[0])
            one_list.append(t.numpy()[0])
        
        # Max Logit Compare
        if zero_max > one_max - 0.001 * self.decay:
            # Return 0
            thread = 0
        else:
            thread = test_threads[one_max_ind]

        return thread
    
    def classify_logits_median(self, all_logits, test_threads):
        """
        Classify Method, logits median. 
        Takes the absolute median value of each class, compares them. 

        Parameters:

            all_logits (torch.Tensor) : Logits from output of model.
            test_threads (list) : List of test threads.

        Returns:

            thread (int) : Classified thread.
        """
        zero_max = -float('inf')
        one_max = -float('inf')
        zero_max_ind = -1
        one_max_ind = -1
        logit = all_logits[0]
        for i, t in enumerate(logit):
            if t[0] > zero_max:
                zero_max = t[0]
                zero_max_ind = i
            if t[1] > one_max:
                one_max = t[1]
                one_max_ind = i
        
        zero_list = []
        one_list = []
        logit = all_logits[0]
        for t in logit:
            zero_list.append(t.numpy()[0])
            one_list.append(t.numpy()[0])

        # Median Logit Compare
        if np.median(zero_list) > (np.median(one_list) - 0.001 * self.decay):
            thread = 0
        else:
            thread = test_threads[one_max_ind]

        return thread
    
    def downsample(self, X_tokens, X_others, y):
        """
        Downsamples based on ratio due to class imbalances.

        Parameters: 

            X_tokens (array-like) : Tokens
            X_others (array-like) : Other features
            y (array-like) : Labels

        Returns:

            df (pd.DataFrame) : Downsampled dataframe
        """

        ones = y.count(1)
        zeros = y.count(0)
        ones_ratio = ones / len(y)
        zeros_ratio = zeros / len(y)
        print('Num Pairs Before: ', len(y))
        print('ones', ones_ratio, 'zeros', zeros_ratio)

        # Combine into single dataframe for splitting
        df = pd.concat([pd.DataFrame(X_tokens), pd.DataFrame(X_others), pd.DataFrame(y)], axis=1) #token_pair[0], token_pair[1], other_pair[0], other_pair[1], label
    

        if ones_ratio > self.desired_ratio or zeros_ratio > self.desired_ratio:
            # In the case that ones > zeros, just need to reduce the ones by sampling
            # In the case that zeros > ones, we want to reduce such that ones becomes majority
            
            # Find the majority class
            if ones_ratio > self.desired_ratio:
                maj_class = 1
                min_ = zeros

            else:
                maj_class = 0
                min_ = ones
                #num_samples_to_keep = int((1 - desired_ratio) * (min_//(desired_ratio)))
            
            num_samples_to_keep = int(self.desired_ratio * (min_//(1-self.desired_ratio)))
    
            majority_class = df[df.iloc[:, -1] == maj_class]
            minority_class = df[df.iloc[:, -1] != maj_class]

            majority_class_downsampled = majority_class.sample(n=num_samples_to_keep, random_state = 100)
            df = pd.concat([majority_class_downsampled, minority_class])

            X_tokens = list(df.iloc[:, 0:2].values)
            X_others = list(df.iloc[:,2:4].to_records(index=False))
            
            y = df.iloc[:, 4].squeeze()

            # Downsampled Ratios
            ones = y.tolist().count(1)
            zeros = y.tolist().count(0)
            ones_ratio = ones / len(y)
            zeros_ratio = zeros / len(y)
            print('Downsampling...', '\n', 'ones', ones_ratio, 'zeros', zeros_ratio)
        
        print('Old length: ', len(df))
        # Reduce the size of the training set, as pairing hundreds of rows is insane
        ones_class = df[df.iloc[:, -1] == 1]
        zeros_class = df[df.iloc[:, -1] == 0]

        # Sample
        sample_val = min(self.train_min, len(df))
        while sample_val > len(ones_class) or sample_val > len(zeros_class):
            sample_val = int(0.9*sample_val)
        ones_class = ones_class.sample(n=sample_val, random_state=100).copy()
        zeros_class = zeros_class.sample(n=sample_val, random_state=100).copy()
        df = pd.concat([ones_class, zeros_class])

        return df

    def compare_bert_test(self, X_test_token_pairs, X_test_others_pairs, test_threads, y_test, X_test_tokens):
        """
        Second stage of Phase 2. 
        Compare is used after Predict, if current message is not a reply to another.
        Uses different methods to classify a message. 

        Parameters:
            
            X_test_token_pairs (list(tuple(str, str))) : Textual test pairs.
            X_test_others_pairs (list(tuple(str, str))) : Test pairs for other features, namely time difference and CID difference.
            test_threads (array-like) : Thread ID values for testing data.
            X_test_tokens (list(str)) : List of all test tokens.
            y_test (array-like) : Testing labels.
            model () : Model used for classification.
            batch_size (int) : The size of each training batch. Must be strictly positive.
            model_name (str) : The name of the BERT model to use. Valid options are 'bert', 'roberta', 'albert', 'modular' and 'custom'. 
            truncation_type (Union[None, str]) : The method of truncating sequences longer than 512 tokens. Valid options are None, 'head-only', 'head-and-tail' and 'tail-only'.

        Returns:

            thread (int) : The classified thread.
            actual (list) : List of actual labels.
            preds (list) : List of predicted labels.

        """
        # After doing the original check for whether message is a reply
        # Pair up target message with messages that come before it
        # Using the model, predict to see if they're similar

        # Given the textual pairs, and the threads for those pairs
        test_threads = test_threads.tolist()

        # Create test based on pairs
        y_test = y_test.reset_index(drop=True)
        actual = []
        # Concatenation of the paired sentences together
        for pair in X_test_token_pairs:
            indices = [X_test_tokens.index(pair[0]), X_test_tokens.index(pair[1])]
            if y_test[indices[0]] == y_test[indices[1]]:
                actual.append(1)
            else:
                actual.append(0)

        # Cast to string
        X_test_token_pairs = [(str(i), str(j)) for i,j in X_test_token_pairs]
            
        # Encoding
        if self.truncation_type in ['head-only', 'tail-only', 'head-and-tail']:
            test_text_encoded = tokenizer_encode(X_test_token_pairs, self.tokenizer, False)
            
            test_text_encoded = truncate(test_text_encoded, self.truncation_type)

        else:
            test_text_encoded = tokenizer_encode(X_test_token_pairs, self.tokenizer, True)

        creator_id_similar = []
        time_diff = []
        
        for i in range(len(X_test_others_pairs)):
            time_diff.append(self.nearest_5_min(abs(X_test_others_pairs[i][0][1] - X_test_others_pairs[i][1][1])))

            if X_test_others_pairs[i][0][0] == X_test_others_pairs[i][1][0]:
                creator_id_similar.append(1)
            else:
                creator_id_similar.append(0)
        
        # Convert to tensors (Consider converting to float)
        categorical_features_test = torch.tensor(creator_id_similar).float().unsqueeze(-1)
        numerical_features_test = torch.tensor(time_diff).float().unsqueeze(-1)
        test_labels = torch.tensor(actual)


        # Create dataset
        test_dataset = create_dataset(test_text_encoded, test_labels, categorical_features_test, numerical_features_test)

        # Create dataloader
        test_loader = create_dataloader(test_dataset, self.batch_size, False)

        # -- Testing --
        all_preds, all_logits = test_model_predict(self.model, self.model_name, test_loader, True, True)
        # all_preds is a list of predictions
        # all_logits is a list of tensors

        # Cannot use decision function like in SVM, can only use logits
        # Should apply softmax first
        
        if self.classify == 'voting1':
            thread = self.classify_voting1(all_preds, test_threads)
        
        elif self.classify == 'voting2':
            thread = self.classify_voting2(all_preds, test_threads)

        elif self.classify == 'logits_max':
            thread = self.classify_logits_max(all_logits, test_threads)
        
        else:
            thread = self.classify_logits_median(all_logits, test_threads)

        if thread == 0:
            self.decay = 0
        else:
            self.decay += 1

        return thread, actual, all_preds

    def predict_and_compare_bert(self):
        """
        Phase 2 code. 
        Firstly, trains the model on training data.
        After data preprocessing, predicts whether a message belongs to a previous thread by checking replies. 
        Then, makes use of the trained classifier to make predictions based on different Classify methods.

        Parameters:     

            None

        Returns:

            test_seen (list) : Test messages with their respective predicted threads.
            X_test_tokens (list) : List of all test tokens.

        """
        
        # Columns needed 
        required_cols = [col for col in ['id', 'content', 'creator_id', 'refs', 'timestamps'] if col not in self.feature_cols]
        if required_cols:
            for col in required_cols:
                self.feature_cols.append(col) 

        
        df = self.pairwise_dataset[self.feature_cols]
        thread_ids = self.pairwise_dataset['thread_id']

        # Ensure first few rows of dataset are part of the training set  
        block_size = 0.2
        X_train_slice = df[:(int(block_size * len(df)))]
        y_train_slice = thread_ids.iloc[:(int(block_size * len(df)))]

        df_rest = df[(int(block_size * len(df))):]
        thread_ids_rest = thread_ids.iloc[(int(block_size * len(df))):]

        # -- Perform Train-Test Split --
        if self.stratify == True:
            X_train, X_val, y_train, y_val = train_test_split(df_rest, 
                                                                thread_ids_rest, 
                                                                test_size = 0.4, 
                                                                random_state=100, 
                                                                stratify = thread_ids_rest
                                                                )
        else:
            X_train, X_val, y_train, y_val = train_test_split(df_rest, 
                                                                thread_ids_rest, 
                                                                test_size = 0.4, 
                                                                random_state=100
                                                                )

        
        X_train = pd.concat([X_train_slice, X_train])
        y_train = pd.concat([y_train_slice, y_train])
        
        # Do pairing before encoding
        # Train on the messages that come before the test message
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        post_train_val_df = val_df.copy()
        
        if self.augmentX is not None:
            train_df['content'] = augmentX_BERT_phase2(train_df['content'], augmentation=self.augmentX)
        # -- Prepare Training Data --
        
        # Get training sentence pairs
        X_train_token_pairs, X_train_others_pairs, y_train_pairs = self.pairSentences(X_train, y_train)
        X_val_token_pairs, X_val_others_pairs, y_val_pairs = self.pairSentences(X_val, y_val)
        val_df = pd.concat([pd.DataFrame(X_val_token_pairs), pd.DataFrame(X_val_others_pairs), pd.DataFrame(y_val_pairs)], axis=1)

        self.desired_ratio = 0.5
        train_df = self.downsample(X_train_token_pairs, X_train_others_pairs, y_train_pairs)

        print('New train_df length: ', len(train_df))

        # Split train_df
        X_train_token_pairs = list((train_df.iloc[i,0], train_df.iloc[i,1]) for i in range(len(train_df)))
        X_train_others_pairs = list(train_df.iloc[:,2:4].to_records(index=False))
        y_train_pairs = train_df.iloc[:, 4].squeeze()
        
        print('Num Train Pairs', len(X_train_token_pairs))
        
        # Cast to string
        X_train_token_pairs = [(str(i), str(j)) for i,j in X_train_token_pairs]

        # Encoding
        if self.truncation_type in ['head-only', 'tail-only', 'head-and-tail']:
            train_text_encoded = tokenizer_encode(X_train_token_pairs, self.tokenizer, False)
            train_text_encoded = truncate(train_text_encoded, self.truncation_type)

        else:
            train_text_encoded = tokenizer_encode(X_train_token_pairs, self.tokenizer, True)

        
        creator_id_similar = []

        time_diff = []
        
        for i in range(len(X_train_others_pairs)):
            time_diff.append(self.nearest_5_min(abs(X_train_others_pairs[i][0][1] - X_train_others_pairs[i][1][1])))

            if X_train_others_pairs[i][0][0] == X_train_others_pairs[i][1][0]:
                creator_id_similar.append(1)
            else:
                creator_id_similar.append(0)
        
        # Convert to tensors (Consider converting to float)
        categorical_features_train = torch.tensor(creator_id_similar).float().unsqueeze(-1)
        numerical_features_train = torch.tensor(time_diff).float().unsqueeze(-1)
        train_labels = torch.tensor(y_train_pairs.tolist())

        # Create dataset
        train_dataset = create_dataset(train_text_encoded, train_labels, categorical_features_train, numerical_features_train)

        # Create dataloader
        train_loader = create_dataloader(train_dataset, self.batch_size, False)

        # -- Prepare Validation Data --
        ones_class = val_df[val_df.iloc[:, -1] == 1]
        zeros_class = val_df[val_df.iloc[:, -1] == 0]

        # Sample
        # Also make sure that sample_val is not larger than the actual number of samples
        sample_val = min(self.val_min, len(val_df))
        while sample_val > len(ones_class) or sample_val > len(zeros_class):
            sample_val = int(0.9*sample_val)
        ones_class = ones_class.sample(n=sample_val, random_state=100).copy()
        zeros_class = zeros_class.sample(n=sample_val, random_state=100).copy()
        val_df = pd.concat([ones_class, zeros_class])

        # Split train_df
        X_val_token_pairs = list((val_df.iloc[i,0], val_df.iloc[i,1]) for i in range(len(val_df)))
        X_val_others_pairs = list(val_df.iloc[:,2:4].to_records(index=False))
        y_val_pairs = val_df.iloc[:, 4].squeeze()

        print('Num Val Pairs', len(X_val_token_pairs))

        # Cast to string
        X_val_token_pairs = [(str(i), str(j)) for i,j in X_val_token_pairs]

        # Encoding
        if self.truncation_type in ['head-only', 'tail-only', 'head-and-tail']:
            val_text_encoded = tokenizer_encode(X_val_token_pairs, self.tokenizer, False)
            val_text_encoded = truncate(val_text_encoded, self.truncation_type)

        else:
            val_text_encoded = tokenizer_encode(X_val_token_pairs, self.tokenizer, True)
        
        creator_id_similar = []
        time_diff = []
        for i in range(len(X_val_others_pairs)):
            time_diff.append(self.nearest_5_min(abs(X_val_others_pairs[i][0][1] - X_val_others_pairs[i][1][1]))) 

            if X_val_others_pairs[i][0][0] == X_val_others_pairs[i][1][0]:
                creator_id_similar.append(1)
            else:
                creator_id_similar.append(0)

        # Convert to tensors (Consider converting to float)
        categorical_features_val = torch.tensor(creator_id_similar).float().unsqueeze(-1)
        numerical_features_val = torch.tensor(time_diff).float().unsqueeze(-1)
        val_labels = torch.tensor(y_val_pairs.tolist())

        # Create dataset
        val_dataset = create_dataset(val_text_encoded, val_labels, categorical_features_val, numerical_features_val)

        # Create dataloader
        val_loader = create_dataloader(val_dataset, self.batch_size, False)

        optimizer = Adam(self.model.parameters(), lr = self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        # -- Training and Validation --
        train_loss_list = []
        val_loss_list = []
        for epoch in range(self.num_epochs):
            train_loss, train_acc = train_model(self.model, self.model_name, optimizer, train_loader, self.criterion, epoch, self.num_epochs, True, True, None)
            
            val_loss, val_acc, val_preds = val_model(self.model, self.model_name, val_loader, self.criterion, epoch, self.num_epochs, True, True)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, '
                f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
            print(val_preds)
            print(val_labels)
            print('Val Accuracy: ', metrics.accuracy_score(val_labels, val_preds))
            print('Val Precision: ', metrics.precision_score(val_labels, val_preds))
            print('Val Recall: ', metrics.recall_score(val_labels, val_preds))
            print('Val F1: ', metrics.f1_score(val_labels, val_preds))
            # model.save_pretrained('../data/phase2_bert_epoch_'+str(epoch)+'_lr_'+str(learning_rate))

        self.post_train_tests_bert_phase2(post_train_val_df)

        loss_df = pd.DataFrame({'train_loss' : train_loss_list, 'val_loss' : val_loss_list})
        loss_df.to_csv('../data/loss_df.csv')


        # -- Predict and Compare --

        # Get prediction dataset
        X_test = self.thread_dataset[self.feature_cols]
        y_test = self.thread_dataset['thread_id']

        X_test_tokens = X_test['content'].tolist()

        # -- Post Train Augmentation --
        # X_test_tokens = augmentX_BERT(X_test_tokens, augmentation="swap")
        
        # X_test.loc[:,'content'] = X_test_tokens

        # PCA
        #X_test_vectorised = pca.transform(X_test_vectorised)
        
        # Need to be able to hold all the 'seen' messages from X_test so that we can look back and compare
        cols = self.feature_cols + ['new_thread_id']

        # Create empty dataframe
        test_seen = pd.DataFrame(columns=cols)

        # Define n in terms of seconds
        n = 43200 # 12 hours

        # -- PREDICT AND COMPARE --

        # -- Predict --
        # Initial prediction to see if message can be classified before using the model
        print('Num Test', len(X_test))
        actual=[]
        predicted=[]
        threads=[]
        thread_number = 1
        X_test = X_test.reset_index()
        self.decay = 0
        for index, message in X_test.iterrows(): # message is a Series

            thread = self.predict(X_test, message)
            """print('Index', index)
            print('Message', message)
            print('Thread', thread)
            print('-------')"""
            # -- Compare --
            if thread is None:             
                # Create pairs 
                #test_df = pd.concat([X_test, y_test], axis=1)
                
                #test_df = test_df[((test_df['timestamps'] < message['timestamps']) & (test_df['timestamps'] > (message['timestamps'] - n)))]
                
                if index == 0:
                    # First message in the test set
                    message['new_thread_id'] = thread_number
                    test_seen.loc[len(test_seen)] = message
                    threads.append(thread_number) #appends 1
                
                    continue  
                
                # Filter messages
                test_seen_filtered = test_seen[((test_seen['timestamps'] < message['timestamps']) & (test_seen['timestamps'] > (message['timestamps'] - n)))]
                X_test_seen_filtered = test_seen_filtered.iloc[:, :-1]
                y_test_seen_filtered = test_seen_filtered.iloc[:, -1]
                y_test_seen_filtered = pd.Series(y_test_seen_filtered)

                #X_test_filtered = test_df.iloc[:, :-1]
                #y_test_filtered = test_df.iloc[:, -1:].squeeze()
                #X_test_token_pairs, X_test_others_pairs, y_test_pairs = pairSentences(X_test_filtered, y_test_filtered)
                
                X_test_token_pairs, X_test_others_pairs = self.pairSentences(test_seen_filtered, y_test_seen_filtered, message, train=False)

                thread, a, p = self.compare_bert_test(X_test_token_pairs, X_test_others_pairs, y_test_seen_filtered, y_test, X_test_tokens)
                
                actual.extend(a)
                predicted.extend(p)
                # New thread
                if thread == 0:
                    thread_number += 1
                    thread = thread_number
                    message['new_thread_id'] = thread_number

                else:
                    message['new_thread_id'] = thread

                test_seen.loc[len(test_seen)] = message
            
            # Thread has been classified
            else:
                message['new_thread_id'] = thread
                test_seen.loc[len(test_seen)] = message

            threads.append(thread)

        print('Last thread no', thread_number)
        """print('Actual Ones Percentage: ', actual.count(1) / len(actual))
        print('Predicted Ones Percentage: ', predicted.count(1) / len(predicted))
        print('Accuracy', metrics.accuracy_score(actual, predicted))
        print('F1', metrics.f1_score(actual, predicted))"""

        test_seen['Predicted'] = threads
        test_seen['Actual'] = y_test.tolist()
        print('Done')
        test_seen.to_csv('../data/test_seen_bert.csv')
        return test_seen, X_test_tokens
    
    def post_train_tests_bert_phase2(self, val_df):
        """
        Conducts post train tests on the Phase 2 BERT model. 
        Invariance tests are adversarial tests that perturb the test data to test the model for robustness.

        Parameters:

            val_df (pd.DatFrame) : Validation data to be perturbed.

        Returns: 

            None
        """
        
        # Perturbations
        syn_aug = naw.SynonymAug(aug_p = self.aug_prob)
        swap_aug = naw.RandomWordAug(action='swap', aug_p = self.aug_prob)
        del_aug = naw.RandomWordAug(action='delete', aug_p = self.aug_prob)
        spell_aug = naw.SpellingAug(aug_p=self.aug_prob)
        keyboard_aug = nac.KeyboardAug(aug_char_p=self.aug_prob)
        split_aug = naw.SplitAug(aug_p=self.aug_prob)
        ant_aug = naw.AntonymAug(aug_p = self.aug_prob)
        char_aug = nac.RandomCharAug(aug_char_p=self.aug_prob)

        X_randomDeletion = pd.Series([del_aug.augment(str(text)) for text in val_df['content']])
        X_randomDeletion = X_randomDeletion.apply(lambda x : ' '.join(x))

        X_randomSynonym = pd.Series([syn_aug.augment(str(text)) for text in val_df['content']])
        X_randomSynonym = X_randomSynonym.apply(lambda x : ' '.join(x))

        X_randomSwap = pd.Series([swap_aug.augment(str(text)) for text in val_df['content']])
        X_randomSwap = X_randomSwap.apply(lambda x : ' '.join(x))

        X_randomSpelling = pd.Series([spell_aug.augment(str(text)) for text in val_df['content']])
        X_randomSpelling = X_randomSpelling.apply(lambda x : ' '.join(x))

        X_randomKeyboard = pd.Series([keyboard_aug.augment(str(text)) for text in val_df['content']])
        X_randomKeyboard = X_randomKeyboard.apply(lambda x : ' '.join(x))

        X_randomSplit = pd.Series([split_aug.augment(str(text)) for text in val_df['content']])
        X_randomSplit = X_randomSplit.apply(lambda x : ' '.join(x))

        X_randomAntonym = pd.Series([ant_aug.augment(str(text)) for text in val_df['content']])
        X_randomAntonym = X_randomAntonym.apply(lambda x : ' '.join(x))

        X_randomChar = pd.Series([char_aug.augment(str(text)) for text in val_df['content']])
        X_randomChar = X_randomChar.apply(lambda x : ' '.join(x))

        # -- Encode into format that can be read by model --
        # Run a loop over everything for code reuse
        ind = 0
        augmentations = ['Random Character Replacement', 'Random Deletion', 'Random Synonym', 'Random Swap', 'Random Spelling', 'Random Keyboard Swap', 'Random Split', 'Random Antonym']
        for augmenter in [del_aug, syn_aug, swap_aug, spell_aug, keyboard_aug, split_aug, ant_aug, char_aug]:
            val_copy = val_df.copy()

            # Augment
            val_copy = val_copy.reset_index()
            X_val_aug = pd.Series([augmenter.augment(str(text)) for text in val_copy['content']])

            val_copy['content'] = X_val_aug
            val_copy.dropna(inplace=True)
            X_val = val_copy.iloc[:, :-1].reset_index()
            y_val = val_copy.iloc[:, -1:].reset_index().squeeze()

            X_val_token_pairs, X_val_others_pairs, y_val_pairs = pairSentencesTest(X_val, y_val, bert=True)

            val_copy = pd.concat([pd.DataFrame(X_val_token_pairs), pd.DataFrame(X_val_others_pairs), pd.DataFrame(y_val_pairs)], axis=1)

            ones_class = val_copy[val_copy.iloc[:, -1] == 1]
            zeros_class = val_copy[val_copy.iloc[:, -1] == 0]

            # Sample
            # Also make sure that sample_val is not larger than the actual number of samples
            sample_val = min(self.val_min, len(val_copy))
            while sample_val > len(ones_class) or sample_val > len(zeros_class):
                sample_val = int(0.9*sample_val)
            ones_class = ones_class.sample(n=sample_val, random_state=100).copy()
            zeros_class = zeros_class.sample(n=sample_val, random_state=100).copy()
            val_copy = pd.concat([ones_class, zeros_class])

            # Split train_df
            X_val_token_pairs = list((val_copy.iloc[i,0], val_copy.iloc[i,1]) for i in range(len(val_copy)))
            X_val_others_pairs = list(val_copy.iloc[:,2:4].to_records(index=False))
            y_val_pairs = val_copy.iloc[:, 4].squeeze()

            print('Num Val Pairs', len(X_val_token_pairs))

            # Cast to string
            X_val_token_pairs = [(str(i), str(j)) for i,j in X_val_token_pairs]
            
            # Encoding
            if self.truncation_type in ['head-only', 'tail-only', 'head-and-tail']:
                val_text_encoded = tokenizer_encode(X_val_token_pairs, self.tokenizer, False)
                val_text_encoded = truncate(val_text_encoded, self.truncation_type)

            else:
                val_text_encoded = tokenizer_encode(X_val_token_pairs, self.tokenizer, True)
            
            creator_id_similar = []
            time_diff = []
            for i in range(len(X_val_others_pairs)):
                time_diff.append(self.nearest_5_min(abs(X_val_others_pairs[i][0][1] - X_val_others_pairs[i][1][1]))) 

                if X_val_others_pairs[i][0][0] == X_val_others_pairs[i][1][0]:
                    creator_id_similar.append(1)
                else:
                    creator_id_similar.append(0)

            # Convert to tensors (Consider converting to float)
            categorical_features_val = torch.tensor(creator_id_similar).float().unsqueeze(-1)
            numerical_features_val = torch.tensor(time_diff).float().unsqueeze(-1)
            val_labels = torch.tensor(y_val_pairs.tolist())

            # Create dataset
            val_dataset = create_dataset(val_text_encoded, val_labels, categorical_features_val, numerical_features_val)

            # Create dataloader
            val_loader = create_dataloader(val_dataset, self.batch_size, False)

            # -- Predictions --
            
            _, _, val_preds = val_model(self.model, self.model_name, val_loader, self.criterion, 1, self.num_epochs, categorical=True, numerical=True)
            print(augmentations[ind] + ' F1: ', metrics.f1_score(val_labels, val_preds))
            ind += 1

def main():
    data = pd.read_csv('../data/cleaned_and_synthesised.csv')
    data = data.dropna()
    #data = data[:60]

    # Split data such that larger set is to train and test the SVM (pairwise classification), smaller set is to test the entire model (thread classification)
    first_set = data['thread_id'].iloc[:int(0.6*len(data))].unique().tolist()
    second_set = data['thread_id'].iloc[int(0.6*len(data)):].unique().tolist()


    for i in first_set:
        if i in second_set:
            second_set.remove(i)

    pairwise_dataset = data[data['thread_id'].isin(first_set)]
    thread_dataset = data[data['thread_id'].isin(second_set)]

    print('Pairwise Size: ', len(pairwise_dataset))
    print('Thread Size: ', len(thread_dataset))


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                num_labels = 2, 
                                                                num_extra_dims = 2) 

    # model = CustomBertForSequenceClassification.from_pretrained('../data/phase2_bert_epoch_0_lr_0.0001', num_labels = 2, num_extra_dims=2)
                                                                
    bert_model = Phase2BERT(pairwise_dataset, 
                            thread_dataset, 
                            model=model, 
                            tokenizer=tokenizer,
                            learning_rate=1e-4,
                            batch_size=16,  
                            num_epochs=3,
                            stratify=False,
                            model_name='custom',
                            truncation_type='head-only',
                            train_min=100,
                            val_min=20)

    test_seen, X_test_tokens = bert_model.predict_and_compare_bert()

    print('1 count: ', (test_seen['Predicted'] == 1).sum())

    predict_eval_bert(test_seen)

if __name__ == "__main__":
    main()

    

