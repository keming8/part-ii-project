import ast
import numpy as np
import pandas as pd
import scipy.sparse
import time
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualise import *
from testing_framework import *
from phase2_eval import *

   
class Phase2SVM():
    def __init__(self, 
                 pairwise_dataset,
                 thread_dataset, 
                 feature_cols = ['id', 'content', 'creator_id', 'tokens', 'refs', 'timestamps'], 
                 vectoriser = TfidfVectorizer(), 
                 kernel = 'linear', 
                 decision_function_shape = 'ovr',
                 stratify = True, 
                 few_shot = 1.0, 
                 augmentX = None, 
                 augmentY = False, 
                 C=1.0,
                 gamma=1,
                 aug_prob=0.1,
                 similarity_threshold=1.0,
                 classify = 'decision_function_median',
                 train_min = 4000,
                 val_min = 1000):
        """
        Initialise Phase 2 code. 

        Parameters:     

            pairwise_dataset (pd.DataFrame) : Dataset used for training and validation. Pairwise Classification.
            thread_dataset (pd.DataFrame) : Dataset used for testing. Thread Classification.
            feature_cols (List[str]) : List of feature columns to be used for feature vectors.
            vectoriser (Union[TfidfVectorizer, CountVectorizer]) : The type of vectoriser used to make put tokens into numerical form. Valid options are TfidfVectorizer or CountVectorizer.
            kernel (str) : Kernel function to use for SVM. Valid options are 'linear', 'rbf', 'sigmoid' or 'poly'.
            decision_function_shape (str) : Defines decision function. Valid options are 'ovo' for One-vs-One or 'ovr' for One-vs-Rest,.
            stratify (bool) : States whether to do a stratified train-test-split or not.
            few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
            augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
            augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
            C (float) : SVM's regularisation parameter. Strength of regularisation is inversely proportional to C. Must be strictly positive.
            gamma (Union[str, float]) : Kernel coefficient for 'rbf', 'sigmoid' and 'poly' kernel functions. Valid options are 'scale', 'auto' or a positive float.
            aug_prob (float) : Probability of pre-train augmentations. Must be strictly between 0.0 and 1.0.
            similarity_threshold (float) : Threshold used for decision function.
            classify (str) : Classify method. Valid options are 'voting1', 'voting2', 'decision_function_max', 'decision_function_median' and 'decision_function_similarity_threshold'.
            train_min (int) : Minimum training samples for each class.
            val_min (int) : Minimum validation samples for each class.

        Returns:

            None

        """
        
        self.pairwise_dataset = pairwise_dataset
        self.thread_dataset = thread_dataset
        self.feature_cols = feature_cols
        self.vectoriser = vectoriser
        self.kernel = kernel
        self.decision_function_shape = decision_function_shape
        self.stratify = stratify
        self.few_shot = few_shot
        self.augmentX = augmentX
        self.augmentY = augmentY
        self.C = C
        self.gamma = gamma
        self.aug_prob = aug_prob
        self.similarity_threshold = similarity_threshold
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

    def pairSentences(self, X_train, y_train=None, X_test=None, y_test=None, train=True, test=False):
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

        # X_train, y_train will be pandas series
        # X_test is a list of tokens
        # y_test is a single label
        
        feature_cols = list(X_train.columns)
        for i in ['id', 'tokens', 'content', 'refs']:
            if i in feature_cols:
                feature_cols.remove(i)
        
        token_pairs = []
        other_pairs = []
        # 1 if same class, 0 if not
        labels = []
        
        if not test:
            X_train_tokens = X_train['tokens'].apply(ast.literal_eval).tolist()
            
        else:
            X_train_tokens = X_train['tokens'].to_list()
        X_train_others = X_train[feature_cols].copy()
        # train_label_list = y_train.tolist()
        if train:
            train_label_list = y_train.tolist()
            for i in range(len(X_train)):
                for j in range(len(X_train)):
                    if i != j:
                        token_pairs.append((X_train_tokens[i], X_train_tokens[j]))
                        other_pairs.append((list(X_train_others.iloc[i].values), list(X_train_others.iloc[j].values)))
                        
                        # Consider adding the reverse here
                        if train_label_list[i] == train_label_list[j]:
                            labels.append(1)
                        else:
                            labels.append(0)
            return token_pairs, other_pairs, labels
        
        elif y_train is not None and X_test is not None:
            feature_cols.remove('new_thread_id')
            X_test_tokens = ast.literal_eval(X_test['tokens'])
            X_test_others = X_test[feature_cols].copy()
            for i in range(len(X_train)):
                token_pairs.append((X_test_tokens, X_train_tokens[i]))
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
    
    def concat_features(self, X_token_pairs, X_others_pairs, X_tokens, X_vectorised):
        """
        Concatenates timestamps and cid differences with the tokens.

        Parameters:

                X_token_pairs (list(tuple)) : Pairs of tokens.
                X_others_pairs (list(tuple)) : Pairs of other features.
                X_tokens (array-like) : All tokens.
                X_vectorised (scipy.sparse) : Sparse matrix of vectorised tokens.

        Returns:

                X_final (pd.DataFrame) : Dataframe with everything concatenated together.
        """

        X_vectorised_pairs = []
        # Concatenation of the paired sentences together
        for pair in X_token_pairs:
            indices = [X_tokens.index(pair[0]), X_tokens.index(pair[1])]
            vectorised_pair = scipy.sparse.hstack([X_vectorised.getrow(indices[0]), X_vectorised.getrow(indices[1])])
            X_vectorised_pairs.append(vectorised_pair)

        X_final = pd.DataFrame(scipy.sparse.vstack(X_vectorised_pairs).toarray())
        
        creator_id_similar = []
        time_diff = []
        for i in range(len(X_others_pairs)):
            time_diff.append(self.nearest_5_min(abs(X_others_pairs[i][0][1] - X_others_pairs[i][1][1]))) # To round off or not?
            if X_others_pairs[i][0][0] == X_others_pairs[i][1][0]:
                creator_id_similar.append(1)
            else:
                creator_id_similar.append(0)

        X_final['creator_id_similar'] = creator_id_similar
        X_final['time_diff'] = time_diff
        X_final.columns = X_final.columns.astype(str)

        return X_final

    
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
    
        # Check for imbalanced dataset
        
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

            majority_class_downsampled = majority_class.sample(n=num_samples_to_keep)
            df = pd.concat([majority_class_downsampled, minority_class])

            X_tokens = list(df.iloc[:, :2].to_records(index=False))
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

    def classify_voting1(self, test_threads, preds):
        """
        Classify Method, voting 1.
        Voting based on the use of predictions from classifier. Takes the thread with the largest count.

        Parameters:

            test_threads (list) : List of test threads.
            preds (list) : List of predictions obtained from classifier.

        Returns:

            thread (int) : Classified thread.
        """

        #Using the predictions and counting from there
        thread_preds = dict()

        assert len(preds) == len(test_threads), 'Preds and labels should have the same length'

        for i in range(len(test_threads)):
            # For new threads
            if preds[i] == 0:
                thread_preds[0] = thread_preds.get(0, 0) + 1
            # For existing threads
            else:
                thread_preds[test_threads[i]] = thread_preds.get(test_threads[i], 0) + 1

        unique_threads = sorted(list(set(thread_preds.keys())))
        for i in range(len(unique_threads)):
            if unique_threads[i] != 0:
                thread_preds[unique_threads[i]] = round(thread_preds[unique_threads[i]] - (self.decay * 0.5))

        # Compare_1 (Voting)
        thread = max(thread_preds.items(), key=lambda item: (item[1], -item[0]))[0]

        return thread
    
    def classify_voting2(self, test_threads, preds):
        """
        Classify Method, voting 2.
        Checks to see if predicted 0s outweigh 1s, before making classifications.

        Parameters:

            test_threads (list) : List of test threads.
            preds (list) : List of predictions obtained from classifier.

        Returns:

            thread (int) : Classified thread.
        """

        #Using the predictions and counting from there
        thread_preds = dict()

        assert len(preds) == len(test_threads), 'Preds and labels should have the same length'

        for i in range(len(test_threads)):
            # For new threads
            if preds[i] == 0:
                thread_preds[0] = thread_preds.get(0, 0) + 1
            # For existing threads
            else:
                thread_preds[test_threads[i]] = thread_preds.get(test_threads[i], 0) + 1

        unique_threads = sorted(list(set(thread_preds.keys())))
        for i in range(len(unique_threads)):
            if unique_threads[i] != 0:
                thread_preds[unique_threads[i]] = round(thread_preds[unique_threads[i]] - (self.decay * 0.5))

        # Compare_2 (Voting)
        if thread_preds.get(0, 0) > sum(thread_preds.values()) - thread_preds.get(0, 0):
            thread = 0
        else:
            if 0 in thread_preds.keys():
                del thread_preds[0]
            thread = max(thread_preds.items(), key=lambda item: (item[1], -item[0]))[0]

        return thread
    
    def classify_decisionfunction_max(self, X_test_final, test_threads):
        """
        Classify Method, decision function max. 
        Takes the absolute max value of each class, compares them. 

        Parameters:

            X_test_final (array-like) : Test data.
            test_threads (list) : List of test threads.

        Returns:

            thread (int) : Classified thread.
        """
        # Using the decision function to get max similarity
        decision_fun = self.classifier.decision_function(X_test_final)

        for i in range(len(decision_fun)):
            decision_fun[i] = decision_fun[i] / (len(decision_fun) - i)
        
        df_max = max(decision_fun)
        df_min = min(decision_fun)

        # Decision Function Compare (Max)
        if abs(df_min) >= (df_max - 0.001 * self.decay):
            # Return 0
            thread = 0
        else:
            # Get the index of df_max
            ind = np.where(decision_fun == df_max)[0][0]
            thread = test_threads[ind]

        return thread
    
    def classify_decisionfunction_median(self, X_test_final, test_threads):
        """
        Classify Method, decision function median. 
        Takes the absoulte median value of each class, compares them. 

        Parameters:

            X_test_final (array-like) : Test data.
            test_threads (list) : List of test threads.

        Returns:

            thread (int) : Classified thread.
        """

        # Using the decision function to get max similarity
        decision_fun = self.classifier.decision_function(X_test_final)

        for i in range(len(decision_fun)):
            decision_fun[i] = decision_fun[i] / (len(decision_fun) - i)
        
        df_max = max(decision_fun)

        positive_scores = decision_fun[decision_fun > 0]
        negative_scores = decision_fun[decision_fun < 0]
        median_pos = np.median(positive_scores) if len(positive_scores) > 0 else 0
        median_neg = np.median(negative_scores) if len(negative_scores) > 0 else 0

        # Decision Function Compare (Median)
        if abs(median_neg) >= (median_pos- 0.001 * self.decay):
            # Return 0
            thread = 0
        else:
            # Get the index of df_max
            ind = np.where(decision_fun == df_max)[0][0]
            thread = test_threads[ind]

        return thread
    
    def classify_decisionfunction_similaritythreshold(self, X_test_final, test_threads):
        """
        Classify Method, decision function similarity threshold. 
        Makes use of the decision function to classify into classes based on the similarity threshold.

        Parameters:

            X_test_final (array-like) : Test data.
            test_threads (list) : List of test threads.

        Returns:

            thread (int) : Classified thread.
        """

        # Using the decision function to get max similarity
        decision_fun = self.classifier.decision_function(X_test_final)

        for i in range(len(decision_fun)):
            decision_fun[i] = decision_fun[i] / (len(decision_fun) - i)
        
        df_max = max(decision_fun)

        # Decision Function Threshold
        if df_max > (self.similarity_threshold - 0.001 * self.decay):
            ind = np.where(decision_fun == df_max)[0][0]
            thread = test_threads[ind]
        else:
            thread = 0

        return thread

    def compare_svm_test(self, X_test_token_pairs, X_test_others_pairs, test_threads, X_test_tokens, X_test_vectorised, y_test):
        """
        Second stage of Phase 2. 
        Compare is used after Predict, if current message is not a reply to another.
        Uses different methods to classify a message. 

        Parameters:
            
            X_test_token_pairs (list(tuple(str, str))) : Textual test pairs.
            X_test_others_pairs (list(tuple(str, str))) : Test pairs for other features, namely time difference and CID difference.
            test_threads (array-like) : Thread ID values for testing data.
            X_test_tokens (list(str)) : List of all test tokens.
            X_test_vectorised (scipy.sparse) : Sparse matrix of vectorised tokens.
            classifier (SVC) : Classifier used for classification.
            y_test (array-like) : Testing labels.
            similarity_threshold (float) : Threshold used for decision function.

        Returns:

            thread (int) : The classified thread.
            actual (list) : List of actual labels.
            preds (list) : List of predicted labels.
        """
        # After doing the original check for whether message is a reply
        # Pair up target message with messages that come before it
        # Using the classifier, classify to see if they're similar

        # Given the textual pairs, and the threads for those pairs
        #print('Compare SVM Test')
        test_threads = test_threads.tolist()
        X_test_vectorised_pairs = []

        actual = []

        y_test = y_test.reset_index(drop=True)
        # Concatenation of the paired sentences together
        for pair in X_test_token_pairs:
            indices = [X_test_tokens.index(pair[0]), X_test_tokens.index(pair[1])]
            if y_test[indices[0]] == y_test[indices[1]]:
                actual.append(1)
            else:
                actual.append(0)
            vectorised_pair = scipy.sparse.hstack([X_test_vectorised.getrow(indices[0]), X_test_vectorised.getrow(indices[1])])
            
            
            X_test_vectorised_pairs.append(vectorised_pair)
    
        if len(X_test_vectorised_pairs) == 0:
            return 0, [], []
        # No pairs, previous message likely to be very far away
        X_test_final = pd.DataFrame(scipy.sparse.vstack(X_test_vectorised_pairs).toarray())

        creator_id_similar = []

        time_diff = []
        for i in range(len(X_test_others_pairs)):
            time_diff.append(self.nearest_5_min(abs(X_test_others_pairs[i][0][1] - X_test_others_pairs[i][1][1]))) # To round off or not?

            if X_test_others_pairs[i][0][0] == X_test_others_pairs[i][1][0]:
                creator_id_similar.append(1)
            else:
                creator_id_similar.append(0)

        X_test_final['creator_id_similar'] = creator_id_similar
        X_test_final['time_diff'] = time_diff
        X_test_final.columns = X_test_final.columns.astype(str)

        preds = self.classifier.predict(X_test_final) #list of 1s and 0s for similar or different

        if self.classify == 'voting1':
            thread = self.classify_voting1(test_threads, preds)

        elif self.classify == 'voting2' : 
            thread = self.classify_voting2(test_threads, preds)

        elif self.classify == 'decision_function_max':
            thread = self.classify_decisionfunction_max(X_test_final, test_threads)

        elif self.classify == 'decision_function_similarity_threshold':
            thread = self.classify_decisionfunction_similaritythreshold(X_test_final, test_threads)

        else:
            thread = self.classify_decisionfunction_median(X_test_final, test_threads)

        if thread == 0:
            self.decay = 0
        else:
            self.decay += 1
        
        return thread, actual, preds

    def predict_and_compare_svm(self):
        """
        Phase 2 code. 
        Firstly, trains the classifier on training data.
        After data preprocessing, predicts whether a message belongs to a previous thread by checking replies. 
        Then, makes use of the trained classifier to make predictions based on different Classify methods.

        Parameters:     

            None

        Returns:

            test_seen (list) : Test messages with their respective predicted threads.
            X_test_vectorised (scipy.sparse) : Sparse matrix of vectorised test values.
            X_test_tokens (list) : List of all test tokens.

        """
        
        # Columns needed 
        required_cols = [col for col in ['id', 'content', 'creator_id', 'tokens', 'refs', 'timestamps'] if col not in self.feature_cols]
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
        if not self.few_shot == 1:
                X_train, _, y_train, _ = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 1 - self.few_shot, 
                                                        random_state = 100
                                                        )

        
        X_train = pd.concat([X_train_slice, X_train])
        y_train = pd.concat([y_train_slice, y_train])

        
        # -- Deal with Tokens --
        X_train_tokens = X_train['tokens'].apply(ast.literal_eval).tolist() #List(List(Str))
        X_val_tokens = X_val['tokens'].apply(ast.literal_eval).tolist()

        # Pre-train augmentations
        if self.augmentX is not None:
            X_train_tokens = augmentX_SVM_phase2(X_train_tokens, self.augmentX)
            
        # Convert to space separated string
        X_train_tokens_str = [' '.join(tokens) for tokens in X_train_tokens]
        X_val_tokens_str = [' '.join(tokens) for tokens in X_val_tokens]
        
        X_train_vectorised = self.vectoriser.fit_transform(X_train_tokens_str)
        X_val_vectorised = self.vectoriser.transform(X_val_tokens_str)

        # Standardisation
        """scaler = StandardScaler(with_mean=False)
        X_train_vectorised = scaler.fit_transform(X_train_vectorised)
        X_val_vectorised = scaler.transform(X_val_vectorised)"""

        # PCA
        """pca = TruncatedSVD(n_components=2)
        X_train_vectorised = pca.fit_transform(X_train_vectorised)
        print(sum(pca.explained_variance_ratio_)) # 90%
        """
        #X_val_vectorised = pca.transform(X_val_vectorised)

        # Try training classifier for each test message
        # Train on the messages that come before the test message
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        post_train_val_df = val_df.copy()
        
        # Filter training set
        X_train_filtered = train_df.iloc[:, :-1]
        y_train_filtered = train_df.iloc[:, -1:].squeeze()
        if self.augmentX is not None:
            X_train_filtered['tokens'] = [str(x) for x in X_train_tokens]
        X_train_filtered.to_csv('../data/val_1.csv')
        X_val_filtered = val_df.iloc[:, :-1]
        y_val_filtered = val_df.iloc[:, -1:].squeeze()


        # -- Prepare Training Data --

        # Get training sentence pairs
        X_train_token_pairs, X_train_others_pairs, y_train_pairs = self.pairSentences(X_train_filtered, y_train_filtered)

        # Check for imbalanced dataset
        self.desired_ratio = 0.5

        train_df = self.downsample(X_train_token_pairs, X_train_others_pairs, y_train_pairs)

        # Split train_df
        X_train_token_pairs = list(train_df.iloc[:, :2].to_records(index=False))
        X_train_others_pairs = list(train_df.iloc[:,2:4].to_records(index=False))
        y_train_pairs = train_df.iloc[:, 4].squeeze()

        # Concatenate paired sentences together
        X_train_final = self.concat_features(X_train_token_pairs, X_train_others_pairs, X_train_tokens, X_train_vectorised)
        X_train_final.to_csv('../data/train.csv')

        # -- Prepare Validation Data --
        X_val_token_pairs, X_val_others_pairs, y_val_pairs = self.pairSentences(X_val_filtered, y_val_filtered)
        val_df = pd.concat([pd.DataFrame(X_val_token_pairs), pd.DataFrame(X_val_others_pairs), pd.DataFrame(y_val_pairs)], axis=1)

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

        # Split val_df
        X_val_token_pairs = list(val_df.iloc[:, :2].to_records(index=False))
        X_val_others_pairs = list(val_df.iloc[:,2:4].to_records(index=False))
        y_val_pairs = val_df.iloc[:, 4].squeeze()

        X_val_final = self.concat_features(X_val_token_pairs, X_val_others_pairs, X_val_tokens, X_val_vectorised)

        # -- Define Classifier --
        self.classifier = svm.SVC(C=self.C, gamma=self.gamma, kernel=self.kernel, decision_function_shape=self.decision_function_shape)
        #classifier = svm.LinearSVC(C=c, max_iter=10000)
        
        if self.augmentY:
            y_train_pairs = augmentY_SVM(y_train_pairs)

        # -- Train Classifier --
        start_time = time.time()
        self.classifier.fit(X_train_final, y_train_pairs)
        end_time = time.time()
        print('Training Time:', end_time - start_time)

        self.post_train_tests_svm_phase2(post_train_val_df)

        # -- Validate Classifier --
        
        val_preds = self.classifier.predict(X_val_final)

        val_df['preds'] = val_preds
        val_df.to_csv('../data/phase2_svm_pairwise.csv')

        print('Predictions: ', val_preds)
        print('Validation Accuracy: ', metrics.accuracy_score(y_val_pairs, val_preds))
        print('Validation Precision: ', metrics.precision_score(y_val_pairs, val_preds))
        print('Validation Recall: ', metrics.recall_score(y_val_pairs, val_preds))
        print('Validation F1: ', metrics.f1_score(y_val_pairs, val_preds))


        # -- Predict and Compare --       

        # Get prediction dataset
        for c in self.feature_cols:
            if c not in self.thread_dataset.columns:
                self.feature_cols.remove(c)

        X_test = self.thread_dataset[self.feature_cols]
        y_test = self.thread_dataset['thread_id']
        # y_test = []

        # Prepare testing data
        X_test_tokens = X_test['tokens'].apply(ast.literal_eval).tolist()

        # -- Post Train Test Augmentation --
        X_test_tokens = augmentX_SVM_phase2(X_test_tokens, augmentation="split")
        X_test.loc[:,'tokens'] = [str(x) for x in X_test_tokens]

        # Convert to space separated string
        X_test_tokens_str = [' '.join(tokens) for tokens in X_test_tokens]

        # Vectorise
        X_test_vectorised = self.vectoriser.transform(X_test_tokens_str)

        # Standardise
        #X_test_vectorised = scaler.transform(X_test_vectorised)

        pd.DataFrame(X_test_vectorised.toarray()).to_csv('../data/X_test_vectorised.csv')
        pd.DataFrame(X_test_tokens).to_csv('../data/X_test_tokens.csv')

        # PCA
        #X_test_vectorised = pca.transform(X_test_vectorised)
        
        # Need to be able to hold all the 'seen' messages from X_test so that we can look back and compare
        cols = self.feature_cols + ['new_thread_id']

        # Create empty dataframe
        test_seen = pd.DataFrame(columns=cols)

        # Define n in terms of seconds
        n = 10800 # 3 hours

        # -- PREDICT AND COMPARE --

        # -- Predict --
        # Initial prediction to see if message can be classified before using the model
        print('Num Test', len(X_test))
        correct_pred = 0
        actual_pred = 0
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

                test_seen_filtered = test_seen[((test_seen['timestamps'] < message['timestamps']) & (test_seen['timestamps'] > (message['timestamps'] - n)))]
                # test_seen_filtered = test_seen[(test_seen['timestamps'] < message['timestamps'])]
                # X_test_seen_filtered = test_seen_filtered.iloc[:, :-1]
                y_test_seen_filtered = test_seen_filtered.iloc[:, -1]
                y_test_seen_filtered = pd.Series(y_test_seen_filtered)

                #X_test_filtered = test_df.iloc[:, :-1]
                #y_test_filtered = test_df.iloc[:, -1:].squeeze()
                #X_test_token_pairs, X_test_others_pairs, y_test_pairs = pairSentences(X_test_filtered, y_test_filtered)
                
                X_test_token_pairs, X_test_others_pairs = self.pairSentences(test_seen_filtered, y_test_seen_filtered, message, train=False)
                
                
                thread, a, p = self.compare_svm_test(X_test_token_pairs, X_test_others_pairs, y_test_seen_filtered, X_test_tokens, X_test_vectorised, y_test)

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
        
        # print('Correct: ', correct_pred)
        # print('Actual: ', actual_pred)
        # print(preds)
        # print(y_test.tolist())
        # print(metrics.accuracy_score(preds, y_test))

        print('Last thread no', thread_number)       
        test_seen['Predicted'] = threads
        test_seen['Actual'] = y_test.tolist()

        print('Done')
        test_seen.to_csv('../data/test_seen_discord_1.csv')
        return test_seen, X_test_vectorised, X_test_tokens
    
    def post_train_tests_svm_phase2(self, val_df):

        """
        Conducts post train tests on the Phase 2 SVM model. 
        Invariance tests are adversarial tests that perturb the test data to test the model for robustness.

        Parameters:

            val_df (pd.DataFrame) : Dataset for testing.

        Returns: 

            None
        """
        # Due to the nature of the data, only the textual data can be modified / perturbed

        prob = self.aug_prob
 
        # Other perturbations using NLP Aug
        syn_aug = naw.SynonymAug(aug_p = prob)
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        spell_aug = naw.SpellingAug(aug_p=prob)
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        split_aug = naw.SplitAug(aug_p=prob)
        ant_aug = naw.AntonymAug(aug_p = prob)
        char_aug = nac.RandomCharAug(aug_char_p=prob)

        augment_list = ['Del', 'Syn', 'Swap', 'Spell', 'Key', 'Split', 'Ant', 'Char']
        ind = 0
        for augmenter in [del_aug, syn_aug, swap_aug, spell_aug, keyboard_aug, split_aug, ant_aug, char_aug]:
            val_copy = val_df.copy()
            # Augment
            val_copy = val_copy.reset_index()
            # X_val_tokens = pd.Series([(augmenter.augment(' '.join(ast.literal_eval(text))) for text in val_df['tokens']).split()])
            X_val_tokens = pd.Series([' '.join(augmenter.augment(' '.join(ast.literal_eval(text)))) for text in val_df['tokens']])   #Series(Str)
            
            val_copy['tokens'] = X_val_tokens
            val_copy.dropna(inplace=True)
            val_copy.set_index('index', inplace=True)
            val_copy.to_csv('../data/val_copy.csv')
            X_val_filtered = val_copy.iloc[:, :-1]
            y_val_filtered = val_copy.iloc[:, -1].squeeze()
            y_val_filtered = pd.DataFrame(y_val_filtered, columns=['thread_id'])
  
            #X_val_filtered['tokens'] = X_val_filtered['tokens'].apply(lambda x : ' '.join(x))
            # Vectorise
            #X_val_tokens_str = [' '.join(tokens) for tokens in X_val_tokens.tolist()]
            X_val_vectorised = self.vectoriser.transform(X_val_tokens)

            # Pair
            X_val_token_pairs, X_val_others_pairs, y_val_pairs = pairSentencesTest(X_val_filtered, y_val_filtered)
            val_copy = pd.concat([pd.DataFrame(X_val_token_pairs), pd.DataFrame(X_val_others_pairs), pd.DataFrame(y_val_pairs)], axis=1)

            ones_class = val_copy[val_copy.iloc[:, -1] == 1]
            zeros_class = val_copy[val_copy.iloc[:, -1] == 0]

            sample_val = min(self.val_min, len(val_copy))
            while sample_val > len(ones_class) or sample_val > len(zeros_class):
                sample_val = int(0.9*sample_val)
            ones_class = ones_class.sample(n=sample_val).copy()
            zeros_class = zeros_class.sample(n=sample_val).copy()
            val_copy = pd.concat([ones_class, zeros_class])

            X_val_token_pairs = list(val_copy.iloc[:, :2].to_records(index=False))
            X_val_others_pairs = list(val_copy.iloc[:,2:4].to_records(index=False))
            y_val_pairs = val_copy.iloc[:, 4].squeeze()
            X_val_tokens = X_val_tokens.tolist()

            X_val_final = self.concat_features(X_val_token_pairs, X_val_others_pairs, X_val_tokens, X_val_vectorised)

            y_pred = self.classifier.predict(X_val_final)

            f1 = metrics.f1_score(y_val_pairs, y_pred, average="macro", zero_division=0)
            
            print(augment_list[ind], 'F1: ', f1)
            ind += 1

def main():
    data = pd.read_csv('../data/cleaned_and_synthesised.csv')
    data_2 = pd.read_csv('../data/discord_test_hacking_cleaned_and_synthesised.csv')

    # Split data such that larger set is to train and test the SVM (pairwise classification), smaller set is to test the entire model (thread classification)
    first_set = data['thread_id'].iloc[:int(0.6*len(data))].unique().tolist()
    second_set = data['thread_id'].iloc[int(0.6*len(data)):].unique().tolist()

    for i in first_set:
        if i in second_set:
            second_set.remove(i)

    pairwise_dataset = data[data['thread_id'].isin(first_set)]
    thread_dataset = data[data['thread_id'].isin(second_set)]
    # thread_dataset = data_2

    phase2svm = Phase2SVM(pairwise_dataset, thread_dataset,  kernel='rbf', stratify=False, C=1, gamma=0.001)
    test_seen, X_test_vectorised, X_test_tokens = phase2svm.predict_and_compare_svm()

    predict_eval(test_seen, X_test_vectorised, X_test_tokens)

    
if __name__ == "__main__":
    main()
