import pandas as pd
import random
import ast
import time
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from visualise import *
from testing_framework import *

class NewGradBoost():
    def __init__(self,
                 data, 
                 n_estimators=100,
                 learning_rate=0.01,
                 max_depth=3,
                 feature_cols = ['tokens'], 
                 vectoriser = TfidfVectorizer(), 
                 oversample = None, 
                 stratify = True, 
                 few_shot = 1, 
                 augmentX = None, 
                 augmentY = False):
        
        """
        Initialise the NewGradBoost Object.

        Parameters:

            data (pd.DataFrame) : Data to be used for classification.
            n_estimators (int) : Number of weak learners to use.
            learning_rate (float) : The size of the step taken during optimisation to update model parameters. Must be strictly positive.
            max_depth (int) : Maximum depth for weak learners.
            feature_cols (List[str]) : List of feature columns to be used for feature vectors.
            vectoriser (Union[TfidfVectorizer, CountVectorizer]) : The type of vectoriser used to make put tokens into numerical form. Valid options are TfidfVectorizer or CountVectorizer.
            oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
            stratify (bool) : States whether to do a stratified train-test-split or not.
            few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
            augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
            augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 

        Returns: 

            None
            
        """
        
         # Check columns
        invalid_cols = [col for col in feature_cols if col not in data.columns]
        if invalid_cols:
            col = invalid_cols[0]
            raise ValueError(f"Invalid columns {col}.")

        # Columns needed for visualisation
        visualise_cols = [col for col in ['content', 'creator_id'] if col not in feature_cols]
        if visualise_cols:
            for col in visualise_cols:
                feature_cols.append(col) 

        self.df = data[feature_cols]
        
        self.data = data
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_cols = feature_cols
        self.vectoriser = vectoriser
        self.oversample = oversample
        self.stratify = stratify
        self.few_shot = few_shot
        self.augmentX = augmentX
        self.augmentY = augmentY
        self.classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    def time_series_gradient_boost(self, time_series_split_n=4):
        """
        Does necessary time series train test split before calling the gradient_boost_helper() function for each iteration.

        Parameters:

            time_series_split_n (int) : Indicates the number of folds to split the data into. 

        Returns: 

            averaged_accuracy (float) : Accuracy score of predictions, averaged over all folds.
            averaged_precision (float) : Precision score of predictions, averaged over all folds.
            averaged_recall (float) : Recall score of predictions, averaged over all folds.
            averaged_f1 (float) : F1 score of predictions, averaged over all folds. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes), only for the last fold.
        """
        
        # If doing time series split
        tscv = TimeSeriesSplit(n_splits=time_series_split_n)
        for train_index, test_index in tscv.split(self.df):
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = self.data['thread_id'].iloc[train_index].ravel(), self.data['thread_id'].iloc[test_index].ravel()

            # -- Few-Shot Learning --
            # Reduce training set size. few_shot to be between 0 and 1
            if not self.few_shot == 1:
                X_train, _, y_train, _ = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 1 - self.few_shot, 
                                                        random_state = 100
                                                        )
            
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            acc, prec, rec, f1, cm = self.gradient_boost_helper(X_train, X_test, y_train, y_test)

            accuracy_list.append(acc)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)

        averaged_accuracy = sum(accuracy_list)/len(accuracy_list)
        averaged_precision = sum(precision_list)/len(precision_list)
        averaged_recall = sum(recall_list)/len(recall_list)
        averaged_f1 = sum(f1_list)/len(f1_list)

        print('Accuracy: ', averaged_accuracy)
        print('Precision: ', averaged_precision)
        print('Recall: ', averaged_recall)
        print('F1 Score: ', averaged_f1)
            
        return averaged_accuracy, averaged_precision, averaged_recall, averaged_f1, cm

    def normal_gradient_boost(self):
        """
        Does necessary train test split before calling the gradient_boost_helper() function.

        Parameters:

            None

        Returns: 

            acc (float) : Accuracy score of predictions.
            prec (float) : Precision score of predictions.
            rec (float) : Recall score of predictions.
            f1 (float) : F1 score of predictions. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes).
        """
        # Perform train-test split
        if self.stratify == True:
            X_train, X_test, y_train, y_test = train_test_split(self.df, 
                                                                self.data['thread_id'], 
                                                                test_size = 0.3, 
                                                                random_state=100, 
                                                                stratify = self.data['thread_id']
                                                                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.df, 
                                                                self.data['thread_id'], 
                                                                test_size = 0.3, 
                                                                random_state=100
                                                                )

        # -- Few-Shot Learning --
        # Reduce training set size. few_shot to be between 0 and 1
        if not self.few_shot == 1:
            X_train, _, y_train, _ = train_test_split(X_train, 
                                                    y_train, 
                                                    test_size = 1 - self.few_shot, 
                                                    random_state = 100
                                                    )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        acc, prec, rec, f1, cm = self.gradient_boost_helper(X_train, X_test, y_train, y_test)
    
        print('Accuracy: ', acc)
        print('Precision: ', prec)
        print('Recall: ', rec)
        print('F1 Score: ', f1)

        return acc, prec, rec, f1, cm

    def gradient_boost_helper(self, X_train, X_test, y_train, y_test):
        """
        Trains and tests GradBoost with specified features. 

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
        if self.augmentX:
            X_train_vectorised = augmentX_SVM(X_train['tokens'], self.vectoriser, augmentation=self.augmentX)
        
        if self.augmentY:
            y_train = augmentY_SVM(y_train)

        # -- Perform concatenation of other features into single feature vector --

        X_train_final = pd.DataFrame(X_train_vectorised.toarray())
        X_test_final = pd.DataFrame(X_test_vectorised.toarray())
    
        # Deal with CIDs and PIDs
        id_col = [col for col in ['binarised', 'binarised_pids', 'all_cids', 'all_pids'] if col in self.feature_cols]
        if id_col:
            for col in id_col:
                X_train_expanded = pd.DataFrame(X_train[col].apply(ast.literal_eval).to_list()).reset_index(drop=True)
                #X_train_expanded = X_train_expanded.map(cleanCell)
                
                X_test_expanded = pd.DataFrame(X_test[col].apply(ast.literal_eval).to_list()).reset_index(drop=True)
                #X_test_expanded = X_test_expanded.map(cleanCell)

                X_train_final = pd.concat([X_train_final, X_train_expanded], axis=1)
                X_test_final = pd.concat([X_test_final, X_test_expanded], axis=1)
        
        # Concatenate the rest of the features into the feature vector
            
        for col in self.feature_cols:
            if col not in ['tokens', 
                        'binarised', 
                        'all_cids', 
                        'binarised_pids', 
                        'all_pids', 
                        'content', 
                        'creator_id'
                        ]:
                X_train_final = pd.concat([X_train_final, X_train[[col]].reset_index(drop=True)], axis = 1)
                X_test_final = pd.concat([X_test_final, X_test[[col]].reset_index(drop=True)], axis = 1)
        
        # Reset column names
        X_train_final.columns = range(len(X_train_final.columns))
        X_test_final.columns = range(len(X_test_final.columns))

        # -- Resampling --
        if self.oversample == 'smote':
            os = SMOTE(k_neighbors=5)
            X_train_final.to_csv('../data/train.csv')
            X_train_final, y_train = os.fit_resample(X_train_final, y_train)

        elif self.oversample == 'duplicate':

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
            majority_class = max(set(y_train), key = y_train_copy.count)
            
            for X, y in combined:
                if class_counts[y] < class_counts[majority_class] // 2:
                    
                    X_train_copy = pd.concat([X_train_copy, X_train_final.iloc[[X]]], axis = 0)
                    y_train_copy.append(y)
                    class_counts[y] += 1

            # Vectorise after resampling
            X_train_final = X_train_copy
            y_train = pd.Series(y_train_copy)

        X_train_final.columns = X_train_final.columns.astype(str)
        X_test_final.columns = X_test_final.columns.astype(str)
        
        # -- Conduct Pre Train Tests --

        pre_train_tests(X_train_final, X_test_final, y_train, y_test)

        # -- Fit the Gradient Boosting Classifier --
        with tqdm(total = len(X_train_final)) as pbar:
            start_time = time.time()
            self.classifier.fit(X_train_final, y_train)
            end_time = time.time()
            pbar.update(len(X_train_final))
            print('Training Time: ', end_time - start_time)
        
        y_pred = self.classifier.predict(X_test_final)

        # -- Conduct Post Train Tests --

        self.post_train_tests_gradboost()
        show_chat(X_test, y_pred)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = metrics.confusion_matrix(y_test, y_pred)


        X_test['Predicted'] = y_pred
        X_test['Actual'] = y_test
        X_test.to_csv('../data/gradboost_test.csv')

        return accuracy, precision, recall, f1, cm
    
    def post_train_tests_gradboost(self):
        """
        Conducts post train tests on the GradBoost classifier. 
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

def new_gradboost(data, 
                  n_estimators=100,
                  learning_rate=0.01,
                  max_depth=3,
                  feature_cols = ['tokens'], 
                  vectoriser = TfidfVectorizer(), 
                  oversample = None, 
                  stratify = True, 
                  few_shot = 1, 
                  augmentX = None, 
                  augmentY = False,
                  time_series_split=False,
                  time_series_split_n=3):
    """
    Helper function to define NewGradBoost object. To be used for trying different settings, features etc. 

    Parameters:

        data (pd.DataFrame) : Data to be used for classification.
        n_estimators (int) : Number of weak learners to use.
        learning_rate (float) : The size of the step taken during optimisation to update model parameters. Must be strictly positive.
        max_depth (int) : Maximum depth for weak learners.
        feature_cols (List[str]) : List of feature columns to be used for feature vectors.
        vectoriser (Union[TfidfVectorizer, CountVectorizer]) : The type of vectoriser used to make put tokens into numerical form. Valid options are TfidfVectorizer or CountVectorizer.
        oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
        stratify (bool) : States whether to do a stratified train-test-split or not.
        few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
        augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
        augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
        time_series_split (bool) : Indicates whether a a time series split is to be done.
        time_series_split_n (int) : Indicates the number of folds for the time series train-test-split. 

    Returns: 

        acc (float) : Accuracy score of predictions.
        prec (float) : Precision score of predictions.
        rec (float) : Recall score of predictions.
        f1 (float) : F1 score of predictions. 
        cm (array-like) : Confusion matrix of shape (n_classes, n_classes).

    """

    classifier = NewGradBoost(data, n_estimators, learning_rate, max_depth, feature_cols, vectoriser, oversample, stratify, few_shot, augmentX, augmentY)

    if time_series_split:
        acc, prec, rec, f1, cm = classifier.time_series_svm(time_series_split_n)
    else:
        acc, prec, rec, f1, cm = classifier.normal_svm()
    
    return acc, prec, rec, f1, cm