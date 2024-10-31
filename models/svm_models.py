import ast
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from typing import Union
from visualise import *
from testing_framework import *


class NewSVM:
    def __init__(self, 
                 data: pd.DataFrame,
                 feature_cols: list[str] = ['tokens'], 
                 vectoriser: Union[TfidfVectorizer, CountVectorizer] = TfidfVectorizer(), 
                 kernel: str = 'linear', 
                 decision_function_shape: str = 'ovr', 
                 oversample: Union[None, str] = None, 
                 stratify: bool = True, 
                 few_shot: float = 1.0, 
                 augmentX: Union[None, str] = None, 
                 augmentY: bool = False, 
                 C: float = 1.0,
                 g: Union[str, float] = 'scale',
                 aug_prob: float = 0.1):
        """
        Initialise the NewSVM Object.

        Parameters:

            data (pd.DataFrame) : Data to be used for classification.
            feature_cols (List[str]) : List of feature columns to be used for feature vectors.
            vectoriser (Union[TfidfVectorizer, CountVectorizer]) : The type of vectoriser used to make put tokens into numerical form. Valid options are TfidfVectorizer or CountVectorizer.
            kernel (str) : Kernel function to use for SVM. Valid options are 'linear', 'rbf', 'sigmoid' or 'poly'.
            decision_function_shape (str) : Defines decision function. Valid options are 'ovo' for One-vs-One or 'ovr' for One-vs-Rest,.
            oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
            stratify (bool) : States whether to do a stratified train-test-split or not.
            few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
            augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
            augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
            C (float) : SVM's regularisation parameter. Strength of regularisation is inversely proportional to C. Must be strictly positive.
            g (Union[str, float]) : Kernel coefficient for 'rbf', 'sigmoid' and 'poly' kernel functions. Valid options are 'scale', 'auto' or a positive float.
            aug_prob (float) : Probability of pre-train augmentations. Must be strictly between 0.0 and 1.0.

        Returns: 

            None
            
        """
        # Check that feature columns are in data
        invalid_cols = [col for col in feature_cols if col not in data.columns]
        if invalid_cols:
            col = invalid_cols[0]
            raise ValueError(f"Invalid columns {col}.")
        
        # Check decision function shape
        if decision_function_shape not in ['ovr', 'ovo']:
            raise ValueError(f"Invalid decision function shape {decision_function_shape}. Please use 'ovo' or 'ovr'.")
        
        # Check kernel
        if kernel not in ['rbf', 'linear', 'poly', 'sigmoid']:
            raise ValueError(f"Invalid kernel function {kernel}. Please use 'rbf', 'linear', 'poly', 'sigmoid'.")
        
        # Check few_shot
        if few_shot < 0.0 or few_shot > 1.0:
            raise ValueError(f"Invalid few shot percentage {few_shot}. Please use a value between 0.0 and 1.0.")
        
        # Check C
        if C < 0.0:
            raise ValueError(f"Invalid value for C {C}. Please use a value greater than 0.0.")
        
        # Check gamma
        if isinstance(g, float):
            if g < 0.0:
                raise ValueError(f"Invalid value for g {g}. Please use a value greater than 0.0.")
        
        self.data = data
        self.df = data[feature_cols]
        self.feature_cols = feature_cols
        self.vectoriser = vectoriser
        self.oversample = oversample
        self.stratify = stratify
        self.few_shot = few_shot
        self.augmentX = augmentX
        self.augmentY = augmentY
        self.aug_prob = aug_prob
        self.classifier = svm.SVC(C=C, gamma=g, kernel = kernel, decision_function_shape=decision_function_shape)

    def few_shot_split(self, X_train, y_train):
        """
        Splits the training data even further for few-shot learning.
        Percentage of training data to keep = self.few_shot.

        Parameters:

            X_train (array-like) : Training input features.
            y_train (array-like) : Training target labels.

        Returns: 

            X_train (array-like) : A percentage of training input features.
            y_train (array-like) : A percentage of training target labels.
        """
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size = 1 - self.few_shot, random_state = 100)

        return X_train, y_train

    def stratified_train_test_split(self, test_size:float =0.3):
        """
        Conducts a stratified train test split where the class ratios of the entire dataset are maintained in the train and test sets.

        Parameters:

            test_size (float) : The size of the test set, in percentage. Must be strictly between 0.0 and 1.0.

        Returns: 

            X_train (array-like) : Training input features.
            X_test (array-like) : Testing input features.
            y_train (array-like) : Training target labels.
            y_test (array-like) : Testing target labels.

        """
        # Check test size
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError(f"Invalid test size {test_size}. Please use a value between 0.0 and 1.0.")
        
        X_train, X_test, y_train, y_test = train_test_split(self.df, 
                                                            self.data['thread_id'], 
                                                            test_size = test_size, 
                                                            random_state=100, 
                                                            stratify = self.data['thread_id']
                                                            )
        return X_train, X_test, y_train, y_test
    
    
    def normal_train_test_split(self, test_size:float =0.3):
        """
        Conducts a normal train test split where the class ratios will be random.

        Parameters:

            test_size (float) : The size of the test set, in percentage. Must be strictly between 0.0 and 1.0.

        Returns: 

            X_train (array-like) : Training input features.
            X_test (array-like) : Testing input features.
            y_train (array-like) : Training target labels.
            y_test (array-like) : Testing target labels.

        """
        # Check test size
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError(f"Invalid test size {test_size}. Please use a value between 0.0 and 1.0.")
        X_train, X_test, y_train, y_test = train_test_split(self.df, 
                                                            self.data['thread_id'], 
                                                            test_size = test_size, 
                                                            random_state=100
                                                            )
        return X_train, X_test, y_train, y_test
    
    def oversample_train(self, X_train_final, y_train):
        """
        Oversamples the training data based on oversampling method self.oversample.

        Parameters:

            X_train_final (array-like) : Input features.
            y_train (array-like) : Target labels.

        Returns: 

            X_train_final (array-like) : Oversampled input features.
            y_train (array-like) : Oversampled target labels based on input features.
        """
        
        if self.oversample == 'smote':
            print('Oversample SMOTE')
            os = SMOTE(k_neighbors=5)
            X_train_final, y_train = os.fit_resample(X_train_final, y_train)

        elif self.oversample == 'duplicate':
            print('Oversample Dupe')

            # Count classes
            class_counts = dict()
            for c in y_train:
                if c in class_counts:
                    class_counts[c] += 1
                else:
                    class_counts[c] = 1
            
            X_train_copy = X_train_final.copy()
            y_train_copy = y_train.copy().tolist()

            combined = list(zip(X_train_final, y_train))
            random.shuffle(combined)
            
            # Find class with most datapoints
            majority_class = max(class_counts, key = lambda k : class_counts[k])
            
            for X, y in combined:
                if class_counts[y] < class_counts[majority_class] // 1.5:
                    
                    X_train_copy = pd.concat([X_train_copy, X_train_final.iloc[[X]]], axis = 0)
                    y_train_copy.append(y)
                    class_counts[y] += 1

            # Vectorise after resampling
            X_train_final = X_train_copy
            y_train = pd.Series(y_train_copy)
        
        else:
            raise ValueError(f"Invalid oversampling method: {self.oversample}. Valid options are 'smote' or 'duplicate'.")

        return X_train_final, y_train
    
    def concat_data_rest(self, X_final, X):
        """
        Concatenates non-list-like features from the dataframe with the feature vector.

        Parameters:

            X_final (array-like) : Processed input features.
            X (array-like) : Original input features.

        Returns: 

            X_final (array-like) : Processed input features, concatenated with non-list-like features.
        """
        for col in self.feature_cols:
            if col not in ['tokens', 
                           'binarised', 
                           'all_cids', 
                           'binarised_pids', 
                           'all_pids', 
                           'content', 
                           'creator_id', 
                           'synthesised_hours_binarised', 
                           'synthesised_day_binarised', 
                           'synthesised_month_binarised', 
                           'synthesised_year_binarised', 
                           'synthesised_dow_binarised',
                           'emojis',
                           'emoticons',
                           'emojis_and_emoticons'
                           ]:
                X_final = pd.concat([X_final, X[[col]].reset_index(drop=True)], axis = 1)
        
        return X_final
    
    def concat_data_list(self, X_final, X):
        """
        Concatenates list-like features from the dataframe with the feature vector.

        Parameters:

            X_final (array-like) : Processed input features.
            X (array-like) : Original input features.

        Returns: 

            X_final (array-like) : Processed input features, concatenated with list-like features.
        """
        id_col = [col for col in ['binarised', 
                                  'binarised_pids', 
                                  'all_cids', 
                                  'all_pids', 
                                  'synthesised_hours_binarised', 
                                  'synthesised_day_binarised', 
                                  'synthesised_month_binarised', 
                                  'synthesised_year_binarised', 
                                  'synthesised_dow_binarised',
                                  'emojis',
                                  'emoticons',
                                  'emojis_and_emoticons'
                                 ] if col in self.feature_cols]
        if id_col:
            for col in id_col:
                X_expanded = pd.DataFrame(X[col].apply(ast.literal_eval).to_list()).reset_index(drop=True)
                X_final = pd.concat([X_final, X_expanded], axis=1)
        
        return X_final
        

    def time_series_svm(self, time_series_split_n : int=5):
        """
        Does necessary time series train test split before calling the svm_helper() function for each iteration.

        Parameters:

            time_series_split_n (int) : Indicates the number of folds to split the data into. 

        Returns: 

            averaged_accuracy (float) : Accuracy score of predictions, averaged over all folds.
            averaged_precision (float) : Precision score of predictions, averaged over all folds.
            averaged_recall (float) : Recall score of predictions, averaged over all folds.
            averaged_f1 (float) : F1 score of predictions, averaged over all folds. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes), only for the last fold.
        """

        tscv = TimeSeriesSplit(n_splits=time_series_split_n)
        for train_index, test_index in tscv.split(self.df):
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = self.data['thread_id'].iloc[train_index].ravel(), self.data['thread_id'].iloc[test_index].ravel()

            # -- Few-Shot Learning --
            # Reduce training set size. few_shot to be between 0 and 1
            if not self.few_shot == 1.0:
                X_train, y_train = self.few_shot_split(X_train, y_train)
            
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            acc, prec, rec, f1, cm = self.svm_helper(X_train, X_test, y_train, y_test)

            accuracy_list.append(acc)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)

        # Metrics for time series split calculated through averaging
        averaged_accuracy = sum(accuracy_list)/len(accuracy_list)
        averaged_precision = sum(precision_list)/len(precision_list)
        averaged_recall = sum(recall_list)/len(recall_list)
        averaged_f1 = sum(f1_list)/len(f1_list)

        print('Accuracy: ', averaged_accuracy)
        print('Precision: ', averaged_precision)
        print('Recall: ', averaged_recall)
        print('F1 Score: ', averaged_f1)
            
        return averaged_accuracy, averaged_precision, averaged_recall, averaged_f1, cm

    def normal_svm(self):
        """
        Does necessary train test split before calling the svm_helper() function.

        Parameters:

            None

        Returns: 

            acc (float) : Accuracy score of predictions.
            prec (float) : Precision score of predictions.
            rec (float) : Recall score of predictions.
            f1 (float) : F1 score of predictions. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes).
        """
        if self.stratify == True:
            X_train, X_test, y_train, y_test = self.stratified_train_test_split()
        else:
            X_train, X_test, y_train, y_test = self.normal_train_test_split()

        # -- Few-Shot Learning --
        # Reduce training set size. few_shot to be between 0 and 1
        if not self.few_shot == 1.0:
            X_train, y_train = self.few_shot_split(X_train, y_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        acc, prec, rec, f1, cm = self.svm_helper(X_train, X_test, y_train, y_test)

        print('Accuracy: ', acc)
        print('Precision: ', prec)
        print('Recall: ', rec)
        print('F1 Score: ', f1)

        return acc, prec, rec, f1, cm
    
    def svm_helper(self, X_train, X_test, y_train, y_test):
        """
        Trains and tests SVM with specified features. 

        Parameters:

            X_train (array-like) : Training input features.
            X_test (array-like) : Testing input features.
            y_train (array-like) : Training target labels.
            y_test (array-like) : Testing target labels.

        Returns: 

            acc (float) : Accuracy score of predictions.
            prec (float) : Precision score of predictions.
            rec (float) : Recall score of predictions.
            f1 (float) : F1 score of predictions. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes).
        """

        # -- Perform Vectorisation --
        X_train_vectorised = self.vectoriser.fit_transform(X_train['tokens'])
        X_test_vectorised = self.vectoriser.transform(X_test['tokens'])

        # -- Add Noise and Perturbations --
        # augmentX can be any of ['synonym', 'word_deletion', 'swap', 'spelling', 'char_replacement']
        if self.augmentX != None:
            X_train_vectorised = augmentX_SVM(X_train['tokens'], y_train, self.vectoriser, augmentation=self.augmentX, aug_prob=self.aug_prob)
        
        if self.augmentY:
            y_train = augmentY_SVM(y_train)

        # -- Perform concatenation of other features into single feature vector --
        X_train_final = pd.DataFrame(X_train_vectorised.toarray())
        X_test_final = pd.DataFrame(X_test_vectorised.toarray())
    
        # Deal with list-like data within dataframe
        X_train_final = self.concat_data_list(X_train_final, X_train)
        X_test_final = self.concat_data_list(X_test_final, X_test)
        
        # Concatenate the rest of the features into the feature vector
        X_train_final = self.concat_data_rest(X_train_final, X_train)
        X_test_final = self.concat_data_rest(X_test_final, X_test)
        
        # Reset column names
        X_train_final.columns = range(len(X_train_final.columns))
        X_test_final.columns = range(len(X_test_final.columns))

        # -- Resampling --
        if self.oversample is not None:
            X_train_final, y_train = self.oversample_train(X_train_final, y_train)

        X_train_final.columns = X_train_final.columns.astype(str)
        X_test_final.columns = X_test_final.columns.astype(str)
        
        # -- Conduct Pre Train Tests --

        pre_train_tests(X_train_final, X_test_final, y_train, y_test)

        # -- Fit the SVM Classifier --
        with tqdm(total = len(X_train_final)) as pbar:
            start_time = time.time()
            self.classifier.fit(X_train_final, y_train)
            end_time = time.time()
            pbar.update(len(X_train_final))
            print('Training Time: ', end_time - start_time)
        y_pred = self.classifier.predict(X_test_final)

        # -- Conduct Post Train Tests --

        self.post_train_tests_svm()
        #show_chat(X_test, y_pred)
        
        X_test['Predicted'] = y_pred
        X_test['Actual'] = y_test
        # X_test.to_csv('../data/phase1_svm.csv')

        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = metrics.confusion_matrix(y_test, y_pred)

        X_test['Predicted'] = y_pred
        X_test['Actual'] = y_test
        X_test.to_csv('../data/svm_test.csv')

        return acc, prec, rec, f1, cm
    
    def post_train_tests_svm(self):
        """
        Conducts post train tests on the SVM classifier. 
        Invariance tests are adversarial tests that perturb the test data to test the model for robustness.
        Distribution tests ensure that the predicted class probabilities sum to 1.

        Parameters:

            None 

        Returns: 

            None
        """

        # Due to the nature of the data, only the textual data can be modified / perturbed
        content = self.X_test['tokens']
        
        # --- Invariance Test ---
        
        # Define Perturbations to be made
        syn_aug = naw.SynonymAug(aug_p = self.aug_prob)
        swap_aug = naw.RandomWordAug(action='swap', aug_p = self.aug_prob)
        del_aug = naw.RandomWordAug(action='delete', aug_p = self.aug_prob)
        spell_aug = naw.SpellingAug(aug_p=self.aug_prob)
        keyboard_aug = nac.KeyboardAug(aug_char_p=self.aug_prob)
        split_aug = naw.SplitAug(aug_p=self.aug_prob)
        ant_aug = naw.AntonymAug(aug_p = self.aug_prob)
        char_aug = nac.RandomCharAug(aug_char_p=self.aug_prob)
        
        augment_list = ['Del', 'Syn', 'Swap', 'Spell', 'Key', 'Split', 'Ant', 'Char']
        ind = 0
        for augmenter in [del_aug, syn_aug, swap_aug, spell_aug, keyboard_aug, split_aug, ant_aug, char_aug]:
        
            X_augmented = pd.Series([augmenter.augment(' '.join(ast.literal_eval(text))) for text in content])
            X_augmented = X_augmented.apply(lambda x : ' '.join(x))
            X_augmented = self.vectoriser.transform(X_augmented)

            X_augmented = pd.DataFrame(X_augmented.toarray())
    
            # -- Perform concatenation of other features into single feature vector --
            X_augmented = self.concat_data_list(X_augmented, self.X_test)

            # Concatenate the rest of the features into the feature vector
            X_augmented = self.concat_data_rest(X_augmented, self.X_test)
                    
            X_augmented.columns = range(len(X_augmented.columns))
            X_augmented.columns = X_augmented.columns.astype(str)

            y_preds = self.classifier.predict(X_augmented)
            f1 = metrics.f1_score(self.y_test, y_preds, average="macro", zero_division=0)
            
            print(augment_list[ind], ' F1: ', f1)
            ind += 1

        X_normal_vectorised = self.vectoriser.transform(self.X_test['tokens'])
        X_normal_vectorised = pd.DataFrame(X_normal_vectorised.toarray())
        X_normal_vectorised= self.concat_data_list(X_normal_vectorised, self.X_test)
        X_normal_vectorised = self.concat_data_rest(X_normal_vectorised, self.X_test)
        X_normal_final = X_normal_vectorised
        
        X_normal_final.columns = range(len(X_normal_final.columns))
        X_normal_final.columns = X_normal_final.columns.astype(str)
            
        # --- Distribution Test ---
        
        decision_scores = self.classifier.decision_function(X_normal_final)
        
        softmax_probs = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
        sums = np.sum(softmax_probs, axis=1)
        
        assert np.allclose(sums, 1.0, atol=1e-6), "Distribution Test Failed"

def new_svm(data: pd.DataFrame,
            feature_cols: list[str] = ['tokens'], 
            vectoriser: Union[TfidfVectorizer, CountVectorizer] = TfidfVectorizer(), 
            kernel: str = 'linear', 
            decision_function_shape: str = 'ovr', 
            oversample: Union[None, str] = None, 
            stratify: bool = True, 
            few_shot: float = 1.0, 
            augmentX: Union[None, str] = None, 
            augmentY: bool = False, 
            C: float = 1.0,
            g: Union[str, float] = 'scale',
            aug_prob: float = 0.1,
            time_series_split: bool = False,
            time_series_split_n:int = 5):
    """
    Helper function to define NewSVM object. To be used for trying different settings, features etc. 

    Parameters:

        feature_cols (List[str]) : List of feature columns to be used for feature vectors.
        vectoriser (Union[TfidfVectorizer, CountVectorizer]) : The type of vectoriser used to make put tokens into numerical form. Valid options are TfidfVectorizer or CountVectorizer.
        kernel (str) : Kernel function to use for SVM. Valid options are 'linear', 'rbf', 'sigmoid' or 'poly'.
        decision_function_shape (str) : Defines decision function. Valid options are 'ovo' for One-vs-One or 'ovr' for One-vs-Rest,.
        oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
        stratify (bool) : States whether to do a stratified train-test-split or not.
        few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
        augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
        augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
        C (float) : SVM's regularisation parameter. Strength of regularisation is inversely proportional to C. Must be strictly positive.
        g (Union[str, float]) : Kernel coefficient for 'rbf', 'sigmoid' and 'poly' kernel functions. Valid options are 'scale', 'auto' or a positive float.
        aug_prob (float) : Probability of pre-train augmentations. Must be strictly between 0.0 and 1.0.
        time_series_split (bool) : Indicates whether a a time series split is to be done.
        time_series_split_n (int) : Indicates the number of folds for the time series train-test-split. 

    Returns: 

        acc (float) : Accuracy score of predictions.
        prec (float) : Precision score of predictions.
        rec (float) : Recall score of predictions.
        f1 (float) : F1 score of predictions. 
        cm (array-like) : Confusion matrix of shape (n_classes, n_classes).

    """
    
    classifier = NewSVM(data, feature_cols, vectoriser, kernel,decision_function_shape, oversample, stratify, few_shot, augmentX, augmentY, C, g, aug_prob)

    if time_series_split:
        acc, prec, rec, f1, cm = classifier.time_series_svm(time_series_split_n)
    else:
        acc, prec, rec, f1, cm = classifier.normal_svm()
    
    return acc, prec, rec, f1, cm