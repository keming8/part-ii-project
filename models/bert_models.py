import pandas as pd
import torch
import torch.nn as nn
import ast
import re
import transformers
from datasets import load_dataset
from imblearn.over_sampling import SMOTE
from multimodal_transformers.model import BertWithTabular, TabularConfig
from sklearn import metrics
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import TensorDataset
from torch.optim import Adam, AdamW, lr_scheduler
from typing import Union
from testing_framework import *
from visualise import *
from bert_helpers import *




class ClassificationHeadRelu(nn.Module):

    def __init__(self, config, num_extra_dims):
        super().__init__()
        # Increase Dimensions to accommodate
        # Categorical and Numerical features
        total_dims = config.hidden_size + num_extra_dims

        # Fully-Connected Layers
        self.linear = nn.Linear(total_dims, total_dims)
        self.proj = nn.Linear(total_dims, config.num_labels)
        classifier_dropout = (config.classifier_dropout 
                              if config.classifier_dropout is not None 
                              else config.hidden_dropout_prob)
        
        # Dropout Layer for Regularisation
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, features):
        x = self.linear(features)
        x = torch.relu(x) # Activation Function
        x = self.dropout(x) # Dropout Layer
        x = self.proj(x)
        return x
    
class ClassificationHeadTanh(nn.Module):

    def __init__(self, config, num_extra_dims):
        super().__init__()
        total_dims = config.hidden_size + num_extra_dims
        self.linear = nn.Linear(total_dims, total_dims)
        self.proj = nn.Linear(total_dims, config.num_labels)
        classifier_dropout = (
                config.classifier_dropout 
                if config.classifier_dropout is not None 
                else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, features):
        x = self.linear(features)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x
"""
The following code edits the code for the BERT model, reference: 
https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_bert.html#BertForSequenceClassification
"""
class CustomBertForSequenceClassification(transformers.BertForSequenceClassification):

    def __init__(self, config, num_extra_dims):

        super().__init__(config)

        self.num_extra_dims = num_extra_dims
        self.num_labels = config.num_labels
        self.config = config
        self.bert =  transformers.BertModel(config)
        classifier_dropout = (
                config.classifier_dropout 
                if config.classifier_dropout is not None 
                else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Classification heads
        self.classifier = ClassificationHeadRelu(config,num_extra_dims) # Custom ReLU Head
        # self.classifier = ClassificationHeadTanh(config,num_extra_dims) # Custom Tanh Head
        # self.classifier = nn.Linear(config.hidden_size + num_extra_dims, config.num_labels) # Traditional linear with extra dims

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        extra_data= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        labels= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Experiment with either last_hidden_layer (outputs[0]) or pooled_output (outputs[1])

        sequence_output = outputs[0] #(batch_size, seq_length (at most 512), hidden_size)
        #sequence_output = outputs[1]
     
        # additional data should be (batch_size, num_extra_dims)
        cls_embedding = sequence_output[:, 0, :] #(batch_size, hidden_size)
        if extra_data is not None:
            output = torch.cat((cls_embedding, extra_data), dim=-1)
        else:
            output = cls_embedding
        output = self.dropout(output)
        
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long 
                                              or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), 
                                                labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class NewBERT():
    def __init__(self, 
                 data: pd.DataFrame, 
                 learning_rate: float, 
                 batch_size: int, 
                 num_epochs: int, 
                 feature_cols: list[str] = ['content', 'binarised', 'creator_id'], 
                 model_name: str = 'bert', 
                 truncation_type: Union[None, str] = None, 
                 pre_train: Union[None, str] = None, 
                 few_shot: float = 1.0, 
                 features_as_text: bool = False, 
                 stratify: bool = True, 
                 oversample: Union[None, str] = None,
                 augmentX: Union[None, str] = None,
                 augmentY: bool = False,
                 optimizer_grouped_params: bool = False):
        
        """
        Initialise the NewBERT Object.

        Parameters:

            data (pd.DataFrame) : Data to be used for classification.
            learning_rate (float) : The size of the step taken during optimisation to update model parameters. Must be strictly positive.
            batch_size (int) : The size of each training batch. Must be strictly positive.
            num_epochs (int) : The number of epochs to train for. Must be strictly positive. 
            feature_cols (List[str]) : List of feature columns to be used for feature vectors.
            model_name (str) : The name of the BERT model to use. Valid options are 'bert', 'roberta', 'albert', 'modular' and 'custom'. 
            truncation_type (Union[None, str]) : The method of truncating sequences longer than 512 tokens. Valid options are None, 'head-only', 'head-and-tail' and 'tail-only'.
            pre_train (Union[None, str]) : Indicates whether pre-training will take place. Valid options are None, 'in-domain' and 'cross-domain'. 
            few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
            features_as_text (bool) : Indicates whether categorical and numerical features are to be concatenated as text for classification.
            stratify (bool) : States whether to do a stratified train-test-split or not.
            oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
            augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
            augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
            optimizer_grouped_params (bool) : Indicates whether to group parameters for discriminate fine-tuning.

        Returns: 

            None
            
        """
        
        # Check columns
        invalid_cols = [col for col in feature_cols if col not in data.columns]
        if invalid_cols:
            col = invalid_cols[0]
            raise ValueError(f"Invalid columns {col}.")
        
        # Check learning rate
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate {learning_rate}. Please use a value greater than 0.0.")
        
        # Check batch size
        if learning_rate < 0:
            raise ValueError(f"Invalid batch size {batch_size}. Please use a value greater than 0.0.")
        
        # Check number of epochs
        if num_epochs < 0:
            raise ValueError(f"Invalid number of epochs {num_epochs}. Please use a value greater than 0.0.")
        
        # Check model name
        if model_name not in ['bert', 'albert', 'roberta', 'modular', 'custom']:
            raise ValueError(f"Invalid model name {model_name}. Valid options are 'bert', 'roberta', 'albert', 'modular' and 'custom'.")
        
        # Check few_shot
        if few_shot < 0.0 or few_shot > 1.0:
            raise ValueError(f"Invalid few shot percentage {few_shot}. Please use a value between 0.0 and 1.0.")

        # Columns needed for visualisation
        visualise_cols = [col for col in ['content', 'creator_id'] if col not in feature_cols]
        if visualise_cols:
            for col in visualise_cols:
                feature_cols.append(col) 
        
        # Columns needed for classification
        class_cols = [col for col in ['content'] if col not in feature_cols]
        if class_cols:
            for col in class_cols:
                feature_cols.append(col) 

        self.data = data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_cols = feature_cols
        self.model_name = model_name
        self.truncation_type = truncation_type
        self.pre_train = pre_train
        self.few_shot = few_shot
        self.features_as_text = features_as_text
        self.stratify = stratify
        self.oversample = oversample
        self.augmentX = augmentX
        self.augmentY = augmentY
        self.optimizer_grouped_parameters = optimizer_grouped_params

    def add_nontextual_features(self):

        # Add Non Textual Features
        self.numerical_cols = [col for col in self.feature_cols if col in ['synthesised_days', 
                                                                      'synthesised_months', 
                                                                      'synthesised_years', 
                                                                      'synthesised_hours', 
                                                                      'NOUN', 'VERB', 'ADJ', 
                                                                      'AUX', 'NUM', 'ADV', 
                                                                      'ADP', 'PRON', 'PART', 
                                                                      'INTJ', 'DET', 'X', 
                                                                      'SCONJ', 'CCONJ', 
                                                                      'PUNCT', 'SYM',
                                                                      'all_cids'
                                                                      ]]

        # Check to see model type. numerical data can only be used with modular BERT for best performance
        if (len(self.numerical_cols) > 0) and self.model_name not in ['modular', 'custom']:
            col = self.numerical_cols[0]
            raise ValueError(f'Only modular or custom BERT can use numerical columns {col}')
        
        self.categorical_cols = [col for col in self.feature_cols if col in ['binarised', 
                                                                        'binarised_pids',
                                                                        'emojis', 
                                                                        'emoticons', 
                                                                        'emojis_and_emoticons',
                                                                        'synthesised_day_binarised',
                                                                        'synthesised_month_binarised',
                                                                        'synthesised_hour_binarised',
                                                                        'synthesised_dow_binarised',
                                                                        'synthesised_year_binarised'
                                                                        ]]
        # Concatenating extra features as text:
        # If want to use single [SEP] token to separate text from categories
        if self.features_as_text == True and self.model_name != 'modular':
            self.data['content'] = self.data.apply(lambda row : concatenate_features_as_text(row, tokens=False), axis=1)

            # Prevents doubling up in custom BERT
            self.categorical_cols = []
            self.numerical_cols = []

        self.thread_ids = self.data['thread_id'].to_list()
        self.thread_id_list = self.data['thread_id'].unique().tolist()

        # Create a new column in dataframe to hold all categorical data
        if (len(self.categorical_cols) > 0) and 'categorical' not in self.data.columns:
            temp_list = []

            for string_list in self.data[self.categorical_cols].values:
                new_list = []
                for s in string_list:
                    new_list.extend(ast.literal_eval(s))
                temp_list.append(new_list)

            self.data['categorical'] = temp_list
            self.feature_cols = [col for col in self.feature_cols if col not in self.categorical_cols]
            self.feature_cols.append('categorical')
        
        # Create a new column in dataframe to hold all numerical data
        if self.numerical_cols and 'numerical' not in self.data.columns:
            temp_list = []

            for string_list in self.data[self.numerical_cols].values:
                new_list = []
                for s in string_list:
                    new_list.append(s)
                temp_list.append(new_list)

            self.data['numerical'] = temp_list
            self.feature_cols = [col for col in self.feature_cols if col not in self.numerical_cols]
            self.feature_cols.append('numerical')

    def get_bert_model(self):
        """
        Chooses the BERT model to use, based on self.model_name

        Parameters:

            None

        Returns: 

            None
        """
        
        # --- Choosing BERT Model ---

        # -- roBERTa --
        if self.model_name == 'roberta':
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
            self.model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base', 
                                                                    num_labels = len(self.thread_id_list)
                                                                    )

        # -- ALBERT --
        elif self.model_name == 'albert':
            self.tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = transformers.AlbertForSequenceClassification.from_pretrained('albert-base-v2', 
                                                                    num_labels = len(self.thread_id_list)
                                                                    )
        
        # -- Custom BERT --
        elif self.model_name == 'custom':
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            if 'categorical' in self.feature_cols and 'numerical' in self.feature_cols:
                extra = len(self.data['categorical'].iloc[0]) + len(self.data['numerical'].iloc[0])
            elif 'categorical' in self.feature_cols:
                extra = len(self.data['categorical'].iloc[0])
            else:
                extra = 0
            self.model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                        num_labels = len(self.thread_id_list), 
                                                                        num_extra_dims = extra 
                                                                        )
        # # -- Modular BERT --
        elif self.model_name == 'modular':
            bert_config = transformers.BertConfig.from_pretrained('bert-base-uncased')

            tabular_config = TabularConfig(
            combine_feat_method='attention_on_cat_and_numerical_feats', 
            cat_feat_dim=len(self.data['categorical'].iloc[0]) if 'categorical' in self.feature_cols else 0,  
            numerical_feat_dim=len(self.numerical_cols),  
            num_labels=len(self.thread_id_list),  
            use_num_bn=False,
            )

            bert_config.tabular_config = tabular_config
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertWithTabular.from_pretrained('bert-base-uncased', 
                                                    config=bert_config
                                                    )

        # -- BERT --
        else:
            config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
            config.hidden_dropout_prob = 0.25
            config.num_labels = len(self.thread_id_list)
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                config=config
                                                                )
    
    def get_optimiser_grouped_parameters(self):
        """
        Gets parameter groups for discriminate fine-tuning.

        Parameters:

            None

        Returns: 

            None
        """
        # -- Group Parameters for Discriminate Fine-Tuning --
        param_list = list(self.model.named_parameters())
        learning_rates = [(self.learning_rate/max(i * 2.6, 1)) for i in range(11, -1, -1)]
        
        if self.model_name in ['albert', 'roberta']:
            mod_name = self.model_name
        else:
            mod_name = 'bert'
        layer_groups = {}
        for n, p in param_list:
            match = re.match(f'{mod_name}\.encoder\.layer\.(\d+)', n)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in layer_groups:
                    layer_groups[layer_num] = []
                layer_groups[layer_num].append(p)
        
        self.optimizer_grouped_parameters = [{'params': layer_params, 'lr': learning_rates[layer_num]} for layer_num, layer_params in layer_groups.items()]

    
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
    
    def oversample_train(self, X_tensors, y_tensors, k_neighbors: int = 5):
        """
        Oversamples the training data based on oversampling method self.oversample.

        Parameters:

            X_tensors (array-like) : Input features.
            y_tensors (array-like) : Target labels.

        Returns: 

            X_train_final (array-like) : Oversampled input features.
            y_train (array-like) : Oversampled target labels based on input features.
        """
        
        if self.oversample == 'smote':
            print('Oversample SMOTE')
            os = SMOTE(k_neighbors=k_neighbors)
            oversampled_data, oversampled_labels = os.fit_resample(X_tensors, y_tensors)
            train_dataset = TensorDataset(*oversampled_data, oversampled_labels)
        
        else:
            raise ValueError(f"Invalid oversampling method: {self.oversample}. Valid options are 'smote' or 'duplicate'.")

        return train_dataset

    
    def time_series_bert(self, time_series_split_n:int = 5):
        """
        Does necessary time series train test split before calling the bert() function for each iteration.

        Parameters:

            time_series_split_n (int) : Indicates the number of folds to split the data into. 

        Returns: 

            averaged_accuracy (float) : Accuracy score of predictions, averaged over all folds.
            averaged_precision (float) : Precision score of predictions, averaged over all folds.
            averaged_recall (float) : Recall score of predictions, averaged over all folds.
            averaged_f1 (float) : F1 score of predictions, averaged over all folds. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes), only for the last fold.
        """

        # Add nontextual features into feature cols
        self.add_nontextual_features()

        # Choose BERT model
        self.get_bert_model()

        # Get optimiser grouped parameters
        if self.optimizer_grouped_parameters:
            self.get_optimiser_grouped_parameters()
        else:
            self.optimizer_grouped_parameters = None

        self.df = self.data[self.feature_cols]

        # -- Time Series Split --
        tscv = TimeSeriesSplit(n_splits=time_series_split_n)

        split_ratio = int(len(self.df) * 0.8)  # 0.2 test
        train_data, X_test = self.df.iloc[:split_ratio], self.df.iloc[split_ratio:]
        train_target, y_test = self.data['thread_id'].iloc[:split_ratio], self.data['thread_id'].iloc[split_ratio:]

        y_test = y_test.tolist()
        
        for train_index, val_index in tscv.split(train_data):
            X_train, X_val = train_data.iloc[train_index], train_data.iloc[val_index]
            y_train, y_val = train_target.iloc[train_index].ravel(), train_target.iloc[val_index].ravel()
            
            y_train = y_train.tolist()
            y_val = y_val.tolist()

            # -- Few-Shot Learning --
            # Reduce training set size. few_shot to be between 0 and 1
            if not self.few_shot == 1.0:
                X_train, y_train = self.few_shot_split(X_train, y_train)

            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            acc, prec, rec, f1, cm = self.bert(X_train, X_val, X_test, y_train, y_val, y_test)

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

        # Drop the categorical and numerical columns from the data so as to not mess up other iterations 
        if 'categorical' in self.data.columns:
            self.data.drop('categorical', axis='columns', inplace=True)

        if 'numerical' in self.data.columns:
            self.data.drop('numerical', axis='columns', inplace=True)
            
        return averaged_accuracy, averaged_precision, averaged_recall, averaged_f1, cm

    def normal_bert(self):
        """
        Does necessary train test split before calling the bert() function.

        Parameters:

            None

        Returns: 

            acc (float) : Accuracy score of predictions.
            prec (float) : Precision score of predictions.
            rec (float) : Recall score of predictions.
            f1 (float) : F1 score of predictions. 
            cm (array-like) : Confusion matrix of shape (n_classes, n_classes).

        """
        # Add nontextual features into feature cols
        self.add_nontextual_features()

        # Choose BERT model
        self.get_bert_model()

        # Get optimiser grouped parameters
        if self.optimizer_grouped_parameters:
            self.optimizer_grouped_parameters = self.get_optimiser_grouped_parameters()
        else:
            self.optimizer_grouped_parameters = None

        self.df = self.data[self.feature_cols]

        # -- Train, Validation, Test split --
        # Train: 60, Val: 20, Test: 20
        if self.stratify == True:
            X_train_temp, X_test, y_train_temp, y_test = self.stratified_train_test_split()
            X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=100, stratify=y_train_temp)
        else:
            X_train_temp, X_test, y_train_temp, y_test = self.stratified_train_test_split()
            X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=100)

        # -- Few-Shot Learning --
            # Reduce training set size. few_shot to be between 0 and 1
            if not self.few_shot == 1.0:
                X_train, y_train = self.few_shot_split(X_train, y_train)
        
        acc, prec, rec, f1, cm = self.bert(X_train, X_val, X_test, y_train, y_val, y_test)
            
        # Drop the categorical and numerical columns from the data so as to not mess up other iterations 
        if 'categorical' in self.data.columns:
            self.data.drop('categorical', axis='columns', inplace=True)

        if 'numerical' in self.data.columns:
            self.data.drop('numerical', axis='columns', inplace=True)

        print('Accuracy: ', acc)
        print('Precision: ', prec)
        print('Recall: ', rec)
        print('F1 Score: ', f1)

        return acc, prec, rec, f1, cm

 
    def bert(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Trains and tests BERT with specified features. 

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

        # For visualisation
        X_visualise = X_test[['content', 'creator_id']]

        X_train_text = X_train['content'].to_list()
        X_val_text = X_val['content'].to_list()
        X_test_text = X_test['content'].to_list()
        
        if self.augmentX is not None:
            X_train_text = augmentX_BERT(X_train_text, augmentation=self.augmentX)

        if self.augmentY:
            y_train = augmentY_SVM(y_train)
        

        if len(self.categorical_cols) > 0:
            X_train_categorical = X_train['categorical'].values.tolist()
            X_val_categorical = X_val['categorical'].values.tolist()
            X_test_categorical = X_test['categorical'].values.tolist()

            categorical_features_train = torch.tensor(X_train_categorical).float()
            categorical_features_val = torch.tensor(X_val_categorical).float()
            categorical_features_test = torch.tensor(X_test_categorical).float()

            categorical = True
        else:
            categorical_features_train = None
            categorical_features_val = None
            categorical_features_test = None

            categorical = False

        if len(self.numerical_cols) > 0:
            X_train_numerical = X_train['numerical'].values.tolist()
            X_val_numerical = X_val['numerical'].values.tolist()
            X_test_numerical = X_test['numerical'].values.tolist()

            numerical_features_train = torch.tensor(X_train_numerical).float()
            numerical_features_val = torch.tensor(X_val_numerical).float()
            numerical_features_test = torch.tensor(X_test_numerical).float()

            numerical = True
        
        else:
            numerical_features_train = None
            numerical_features_val = None
            numerical_features_test = None

            numerical = False

        # -- Conduct Pre Train Tests --
        pre_train_tests(X_train, X_test, y_train, y_test)

        train_labels = torch.tensor(y_train.values)
        train_labels = torch.tensor(mapLabels(train_labels.tolist(), createLabelMapping(self.thread_id_list)))  

        val_labels = torch.tensor(y_val.values)
        val_labels = torch.tensor(mapLabels(val_labels.tolist(), createLabelMapping(self.thread_id_list)))

        test_labels = torch.tensor(y_test.values)
        test_labels_mapped = mapLabels(test_labels.tolist(), createLabelMapping(self.thread_id_list)) 

        # -- Truncation and Encoding --
        if self.truncation_type in ['head-only', 'tail-only', 'head-and-tail']:
            train_text_encoded = tokenizer_encode(X_train_text, self.tokenizer, False)
            train_text_encoded = truncate(train_text_encoded, self.truncation_type)

            val_text_encoded = tokenizer_encode(X_val_text, self.tokenizer, False)
            val_text_encoded = truncate(val_text_encoded, self.truncation_type)

            test_text_encoded = tokenizer_encode(X_test_text, self.tokenizer, False)
            test_text_encoded = truncate(test_text_encoded, self.truncation_type)

        else: # Standard truncation
            train_text_encoded = tokenizer_encode(X_train_text, self.tokenizer, True)

            val_text_encoded = tokenizer_encode(X_val_text, self.tokenizer, True)
        
            test_text_encoded = tokenizer_encode(X_test_text, self.tokenizer, True)
        
        # For separating the actual text from the added features
        if self.features_as_text == True and self.model_name != 'modular':
            train_text_encoded['token_type_ids'] = create_token_type_ids(train_text_encoded, self.tokenizer.sep_token_id)

            val_text_encoded['token_type_ids'] = create_token_type_ids(val_text_encoded, self.tokenizer.sep_token_id)

            test_text_encoded['token_type_ids'] = create_token_type_ids(test_text_encoded, self.tokenizer.sep_token_id)
            

        
        # Create DataLoader. This helps with batching, shuffling (randomness for training phase)
        if self.model_name == 'modular' and not(categorical) and not(numerical):
            raise ValueError("Modular BERT is meant to be used with other non textual features. Use regular BERT if no other features.")
        
        train_dataset = create_dataset(train_text_encoded, train_labels, categorical_features_train, numerical_features_train)

        val_dataset = create_dataset(val_text_encoded, val_labels, categorical_features_val, numerical_features_val)

        test_dataset = create_dataset(test_text_encoded, test_labels, categorical_features_test, numerical_features_test)

        # -- Resampling on Dataset --
        if self.oversample is not None:
            train_dataset = self.oversample_train(train_dataset.tensors[:-1], train_dataset.tensors[-1])

        # Put into Data Loader
        train_loader = create_dataloader(train_dataset, self.batch_size, False)

        val_loader = create_dataloader(val_dataset, self.batch_size, False)

        test_loader = create_dataloader(test_dataset, self.batch_size, False)

        # Optimise, can try different ones
        # Change LR
        
        if self.optimizer_grouped_parameters:
            optimizer = AdamW(self.optimizer_grouped_parameters, lr = self.learning_rate)
        else:
            optimizer = Adam(self.model.parameters(), lr = self.learning_rate)
            #optimizer = AdamW(self.model.parameters(), lr = self.learning_rate)

        # Scheduler for learning rate, can try different ones. num_warmup_steps = 0 means no warmup phase
        #num_training_steps = num_epochs * (len(train_dataset) // batch_size)
        #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.1 * num_training_steps, num_training_steps=num_training_steps)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        scheduler=None

        criterion = torch.nn.CrossEntropyLoss()

        if self.pre_train in ['in-domain', 'cross-domain']:
            if self.model_name == 'custom':
                if self.model.num_extra_dims != 0:
                    raise ValueError("Pre Training for custom model with extra features is not supported.")
                
            elif self.model_name == 'modular':
                raise ValueError("Pre Training for modular BERT model is not supported.")

            
            if self.pre_train == 'cross-domain':

                # -- Pre Training Multiclass Classification on AG News Dataset --

                pretrain_data = load_dataset('ag_news')['train']
                pretrain_val_data = load_dataset('ag_news')['test']
                # Limit to 1000 examples, original dataset has 120,000
                pretrain_data = pretrain_data[:1000]
                pretrain_val_data = pretrain_val_data[:1000]

                pretrain_text = pretrain_data['text']
                pretrain_labels = pretrain_data['label']

                pretrain_val_text = pretrain_val_data['text']
                pretrain_val_labels = pretrain_val_data['label']

                pretrain_labels = torch.tensor(pretrain_labels)
                pretrain_val_labels = torch.tensor(pretrain_val_labels)
            
            elif self.pre_train == 'in-domain':

                # -- Pre Training Multiclass Classification on other Hacking forums --
                pretrain_data = pd.read_csv('../data/hacking_combined.csv')

                pretrain_thread_ids = pretrain_data['thread_id'].to_list()
                pretrain_thread_id_list = pretrain_data['thread_id'].unique().tolist()
                
                pretrain_text, pretrain_val_text, pretrain_labels, pretrain_val_labels = train_test_split(pretrain_data['tokens'], pretrain_thread_ids, test_size = 0.3, random_state=100)
                pretrain_text = pretrain_text.to_list()
                pretrain_val_text = pretrain_val_text.to_list()
                
                pretrain_labels = torch.tensor(pretrain_labels)
                pretrain_labels = torch.tensor(mapLabels(pretrain_labels.tolist(), createLabelMapping(pretrain_thread_id_list)))   

                pretrain_val_labels = torch.tensor(pretrain_val_labels)
                pretrain_val_labels_mapped = mapLabels(pretrain_val_labels.tolist(), createLabelMapping(pretrain_thread_id_list))

        
            pretrain_text_encoded = tokenizer_encode(pretrain_text, self.tokenizer, True)

            pretrain_val_text_encoded = tokenizer_encode(pretrain_val_text, self.tokenizer, True)
            
            # -- Create Pretrain Datasets -- 

            pretrain_dataset = create_dataset(pretrain_text_encoded, pretrain_labels)

            pretrain_val_dataset = create_dataset(pretrain_val_text_encoded, pretrain_val_labels)

            # -- Create Pretrain Dataloaders --

            pretrain_loader = create_dataloader(pretrain_dataset, self.batch_size, False)

            pretrain_val_loader = create_dataloader(pretrain_val_dataset, self.batch_size, False)

            # -- Training --

            train_model(self.model, self.model_name, optimizer, pretrain_loader, criterion, self.num_epochs, categorical, numerical)

            # -- Testing --

            all_preds = test_model(self.model, self.model_name, pretrain_val_loader, categorical, numerical)

            if self.pre_train == 'in-domain':
                pretrain_val_labels = pretrain_val_labels_mapped

            print('Pre-Train Accuracy: ', metrics.accuracy_score(pretrain_val_labels, all_preds))
            print('Pre-Train Precision: ', metrics.precision_score(pretrain_val_labels, all_preds, average='macro', zero_division=0))
            print('Pre-Train Recall: ', metrics.recall_score(pretrain_val_labels, all_preds, average='macro', zero_division=0))
            print('Pre-Train F1: ', metrics.f1_score(pretrain_val_labels, all_preds, average='macro', zero_division=0))
        
        
        # -- Training --
        train_loss_list = []
        val_loss_list = []
        for epoch in range(self.num_epochs):
            train_loss, train_acc = train_model(self.model, self.model_name, optimizer, train_loader, criterion, epoch, self.num_epochs, categorical, numerical, scheduler)
            val_loss, val_acc, _ = val_model(self.model, self.model_name, val_loader, criterion, epoch, self.num_epochs, categorical, numerical)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, '
                f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
            

        loss_df = pd.DataFrame({'train_loss' : train_loss_list, 'val_loss' : val_loss_list})
        loss_df.to_csv('../data/loss_df.csv')

        # -- Testing --
        all_preds, all_probs = test_model(self.model, self.model_name, test_loader, categorical, numerical)

        # -- Conduct Post Train Tests --
        self.post_train_tests_bert(X_test, y_test)
        self.distribution_test_bert(all_probs)

        # -- Show chat --
        
        show_chat(X_visualise, all_preds)

        # -- Save Model Weights --
        #torch.save(model.state_dict(), f"../models/bert_models/{self.learning_rate}_{self.batch_size}_{self.num_epochs}.pth")

        # Calculate accuracy on the validation set
        accuracy = metrics.accuracy_score(test_labels_mapped, all_preds)
        precision = metrics.precision_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)
        recall = metrics.recall_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)
        f1 = metrics.f1_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)
        cm = metrics.confusion_matrix(test_labels_mapped, all_preds)

        X_test['Predicted'] = all_preds
        X_test['Actual'] = y_test
        X_test.to_csv('../data/bert_test.csv')


        # Write results to text file
        with open('../data/bert_hyperparams.txt', 'a') as file:
            # Write content to the file
            file.write('\n')
            file.write('\n')
            file.write('Model: ' + self.model_name)
            file.write('\n')
            file.write('Features: ' + ', '.join(self.feature_cols))
            file.write('\n')
            file.write('Learning Rate: ' + str(self.learning_rate))
            file.write('\n')
            file.write('Batch Size: ' + str(self.batch_size))
            file.write('\n')
            file.write('Num Epochs: ' + str(self.num_epochs))
            file.write('\n')
            file.write('Accuracy: ' + str(accuracy))
            file.write('\n')
            file.write('Precision: ' + str(precision))
            file.write('\n')
            file.write('Recall: ' + str(recall))
            file.write('\n')
            file.write('F1 Score: ' + str(f1))
            file.write('\n')
            file.write('\n')
            file.write('---')

        return accuracy, precision, recall, f1, cm
    
    def post_train_tests_bert(self, X_test, y_test):
        """
        Conducts post train tests on the BERT model. 
        Invariance tests are adversarial tests that perturb the test data to test the model for robustness.

        Parameters:

            X_test (array-like) : Input features.
            y_test (array-like) : Input feature labels. 

        Returns: 

            None
        """
        test_labels = torch.tensor(y_test.tolist())
        test_labels_mapped = mapLabels(test_labels.tolist(), createLabelMapping(self.thread_id_list))

        if 'categorical' in X_test.columns:
            X_test_categorical = X_test['categorical'].values.tolist()
            categorical_features_test = torch.tensor(X_test_categorical).float()
            categorical = True
        else:
            categorical_features_test = None

            categorical = False
        
        if 'numerical' in X_test.columns:
            X_test_numerical = X_test['numerical'].values.tolist()
            numerical_features_test = torch.tensor(X_test_numerical).float()
            numerical = True
        
        else:
            numerical_features_test = None

            numerical = False

        #X_test_cid =[ast.literal_eval(l) for l in bin_test] 
        X_test = X_test['content']
        
        prob = 0.1
        
        # Other perturbations using NLP Aug
        syn_aug = naw.SynonymAug(aug_p = prob)
        swap_aug = naw.RandomWordAug(action='swap', aug_p = prob)
        del_aug = naw.RandomWordAug(action='delete', aug_p = prob)
        spell_aug = naw.SpellingAug(aug_p=prob)
        keyboard_aug = nac.KeyboardAug(aug_char_p=prob)
        split_aug = naw.SplitAug(aug_p=prob)
        ant_aug = naw.AntonymAug(aug_p = prob)
        char_aug = nac.RandomCharAug(aug_char_p=prob)

        X_randomDeletion = pd.Series([del_aug.augment(str(text)) for text in X_test])
        X_randomDeletion = X_randomDeletion.apply(lambda x : ' '.join(x))

        X_randomSynonym = pd.Series([syn_aug.augment(str(text)) for text in X_test])
        X_randomSynonym = X_randomSynonym.apply(lambda x : ' '.join(x))

        X_randomSwap = pd.Series([swap_aug.augment(str(text)) for text in X_test])
        X_randomSwap = X_randomSwap.apply(lambda x : ' '.join(x))

        X_randomSpelling = pd.Series([spell_aug.augment(str(text)) for text in X_test])
        X_randomSpelling = X_randomSpelling.apply(lambda x : ' '.join(x))

        X_randomKeyboard = pd.Series([keyboard_aug.augment(str(text)) for text in X_test])
        X_randomKeyboard = X_randomKeyboard.apply(lambda x : ' '.join(x))

        X_randomSplit = pd.Series([split_aug.augment(str(text)) for text in X_test])
        X_randomSplit = X_randomSplit.apply(lambda x : ' '.join(x))

        X_randomAntonym = pd.Series([ant_aug.augment(str(text)) for text in X_test])
        X_randomAntonym = X_randomAntonym.apply(lambda x : ' '.join(x))

        X_randomChar = pd.Series([char_aug.augment(str(text)) for text in X_test])
        X_randomChar = X_randomChar.apply(lambda x : ' '.join(x))

        # -- Encode into format that can be read by model --

        # Run a loop over everything for code reuse
        
        augmentations = ['Normal', 'Random Character Replacement', 'Random Deletion', 'Random Synonym', 'Random Swap', 'Random Spelling', 'Random Keyboard Swap', 'Random Split', 'Random Antonym']
        for i, data in enumerate([X_test, X_randomChar, X_randomDeletion, X_randomSynonym, X_randomSwap, X_randomSpelling, X_randomKeyboard, X_randomSplit, X_randomAntonym]):

            # -- Encode --
            data = data.tolist()
            encoded_X = tokenizer_encode(data, self.tokenizer, True)
        
            if self.features_as_text == True and self.model_name != 'modular':
                encoded_X['token_type_ids'] = create_token_type_ids(encoded_X, self.tokenizer.sep_token_id)

            # -- Create Dataset --
            test_dataset = create_dataset(encoded_X, test_labels, categorical_features_test, numerical_features_test)

            # -- Create Dataloader --
            test_dataloader = create_dataloader(test_dataset, self.batch_size, False)

            # -- Predictions --
            all_preds, all_probs = test_model(self.model, self.model_name, test_dataloader, categorical, numerical)

            accuracy = metrics.accuracy_score(test_labels_mapped, all_preds)
            precision = metrics.precision_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)
            recall = metrics.recall_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)
            f1 = metrics.f1_score(test_labels_mapped, all_preds, average = 'macro', zero_division = 0)

            print(augmentations[i] + ' Accuracy: ', accuracy)
            print(augmentations[i] + ' Precision: ', precision)
            print(augmentations[i] + ' Recall: ', recall)
            print(augmentations[i] +  ' F1: ', f1)
            
    def distribution_test_bert(self, all_probs):
        """
        Conducts distribution tests on the BERT model.
        Distribution tests ensure that the predicted class probabilities sum to 1.

        Parameters:

            all_probs (array-like) : The probabilities of each class, to be summed up.

        Returns:

            None
        """

        sums = np.sum(all_probs, axis=1)
        
        assert np.allclose(sums, 1.0, atol=1e-6), "Distribution Test Failed" 


def new_bert(data: pd.DataFrame, 
             learning_rate: float, 
             batch_size: int, 
             num_epochs: int, 
             feature_cols: list[str] = ['content', 'binarised', 'creator_id'], 
             model_name: str = 'bert', 
             truncation_type: Union[None, str] = None, 
             pre_train: Union[None, str] = None, 
             few_shot: float = 1.0, 
             features_as_text: bool = False, 
             stratify: bool = True, 
             oversample: Union[None, str] = None,
             augmentX: Union[None, str] = None,
             augmentY: bool = False,
             optimizer_grouped_params: bool = False,
             time_series_split: bool = False,
             time_series_split_n:int = 5):
    """
    Helper function to define NewBERT object. To be used for trying different settings, features etc. 

    Parameters:

        data (pd.DataFrame) : Data to be used for classification.
        learning_rate (float) : The size of the step taken during optimisation to update model parameters. Must be strictly positive.
        batch_size (int) : The size of each training batch. Must be strictly positive.
        num_epochs (int) : The number of epochs to train for. Must be strictly positive. 
        feature_cols (List[str]) : List of feature columns to be used for feature vectors.
        model_name (str) : The name of the BERT model to use. Valid options are 'bert', 'roberta', 'albert', 'modular' and 'custom'. 
        truncation_type (Union[None, str]) : The method of truncating sequences longer than 512 tokens. Valid options are None, 'head-only', 'head-and-tail' and 'tail-only'.
        pre_train (Union[None, str]) : Indicates whether pre-training will take place. Valid options are None, 'in-domain' and 'cross-domain'. 
        few_shot (float) : Indicates the percentage of training data to use. Must be strictly between 0.0 and 1.0. 
        features_as_text (bool) : Indicates whether categorical and numerical features are to be concatenated as text for classification.
        stratify (bool) : States whether to do a stratified train-test-split or not.
        oversample (Union[None, str]) : Oversampling method for training data. Valid options are None, 'smote' or 'duplicate'. 
        augmentX (Union[None, str]) : Defines the pre-train X-augmentation to use. Valid options are None, 'synonym', 'word_deletion', 'swap', 'split', 'spelling', 'char_replacement', 'keyboard' or 'antonym'. 
        augmentY (bool) : Indicates whether to do pre-train y-augmentation or not. 
        optimizer_grouped_params (bool) : Indicates whether to group parameters for discriminate fine-tuning.
        time_series_split (bool) : Indicates whether a a time series split is to be done.
        time_series_split_n (int) : Indicates the number of folds for the time series train-test-split. 

    Returns: 

        acc (float) : Accuracy score of predictions.
        prec (float) : Precision score of predictions.
        rec (float) : Recall score of predictions.
        f1 (float) : F1 score of predictions. 
        cm (array-like) : Confusion matrix of shape (n_classes, n_classes).
        
    """
    
    model = NewBERT(data, learning_rate, batch_size, num_epochs, feature_cols, model_name, truncation_type, pre_train, few_shot, 
                    features_as_text, stratify, oversample, augmentX, augmentY, optimizer_grouped_params)
    
    if time_series_split:
        acc, prec, rec, f1, cm = model.time_series_bert(time_series_split_n)
    else:
        acc, prec, rec, f1, cm = model.normal_bert()
    
    return acc, prec, rec, f1, cm