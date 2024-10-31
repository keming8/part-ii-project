import ast
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Union
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer


def createLabelMapping(labelList: list[int]):
    """
    Creates a mapping of thread IDs to an index from 0 to len(labelList).
    Used to map the larger thread ID numbers to small values.

    Parameters:

        labelList (list(int)) : List of thread IDs

    Returns:

        newdict (dict) : A dictionary mapping thread IDs to an integer specifying their index in label list.
    """
    newdict = dict()
    for i in range(len(labelList)):
        newdict[labelList[i]] = i
    return newdict

def mapLabels(labelList: list[int], labelMapping: dict):
    """
    Taking a mapping and a list of labels, maps thread IDs to their respective values based on the mapping.

    Parameters:

        labelList (list(int)) : List of thread IDs
        labelMapping (dict) : Mapping of thread IDs to their new values

    Returns:

        newlist (list(int)) : New list of thread IDs, mapped to their new values.
    """
    newList = []
    for label in labelList:
        newList.append(labelMapping[label])
    return newList


def truncate(encoded_dict: dict, truncation_type: str):
    """
    Truncates sequences that are longer than 512 in 3 different ways.

    Parameters:

        encoded_dict (dict) : Dictionary of indices to tensors, where tensors may be longer than 512 tokens.
        truncation_type (str) : Specifies the type of truncation. Valid options are 'head-only', 'tail-only' and 'head-and-tail'.

    Returns:

        encoded_dict (dict) : Dictionary of indices to tensors, where tensors will be at most 512 tokens long.
    """
    for i in encoded_dict.keys():
        new_tensor = encoded_dict[i]
    
        if truncation_type == 'head-only':
            new_tensor = new_tensor[:, :512]
        elif truncation_type == 'tail-only':
            new_tensor = torch.cat((new_tensor[:, :1], new_tensor[:,-511:]), dim=1)
        else: #head and tail
            new_tensor = torch.cat((new_tensor[:, :130], new_tensor[:, -382:]), dim=1)
        # else: #head and tail
        #     new_tensor = torch.cat((new_tensor[:, :382], new_tensor[:, -130:]), dim=1)
        # else: #head and tail
        #     new_tensor = torch.cat((new_tensor[:, :256], new_tensor[:, -256:]), dim=1)
        
        encoded_dict[i] = new_tensor
    
    return encoded_dict

def concatenate_features_as_text(row, tokens = True):
    """
    Concatenates numerical and categorical features as text

    Parameters:

        row (array-like) : Provides the data containing all features.
        tokens (bool) : Specifies whether to return tokens or strings

    Returns:

        row[col] (array-like) : Returns the data, with categorical and numerical features appended as text. 
    """

    # Need to concatenate PID and/or time

    # Concatenate PIDs
    
    # Extract only the CIDs and not the 0s
    new_cids = []
    cids = ast.literal_eval(row['all_cids'])
    for cid in cids:
        if cid != '0':
            new_cids.append(cid)
    if not tokens:
        col = 'content'
        row['content'] += ' [SEP] '
        row['content'] += ' '.join([str(cid) for cid in new_cids])
        """row['content'] += ' '.join(' [SEP] ' + row['synthesised_day_of_week_str'] + ' [SEP] ' + str(row['synthesised_days'])
                                + ' [SEP] ' + str(row['synthesised_months']) + ' [SEP] ' + str(row['synthesised_years']))"""
    else:
        col = 'tokens'
        token_row = ast.literal_eval(row['tokens'])
        token_row.append('[SEP]')
        for cid in new_cids:
            token_row.append(str(cid))
        row['tokens'] = token_row

    return row[col]

def getCIDs(data):
    """
    Extracts CIDs and puts them into a set.

    Parameters:

        data (array-like) : The dataset, with an 'all_cids' column.

    Returns:

        cid_set : The set of CIDs for that particular entry.
    """

    cids = data['all_cids']

    cid_set = set()

    for cid_list in cids:
        for cid in ast.literal_eval(cid_list):
            cid_set.add(cid)
    
    return cid_set


def create_token_type_ids(encoded_tensor: torch.Tensor, sep_token_id: int):
    """
    Creates new token type IDs for features appended as text.
    Each type of feature will have its own token type ID.

    Parameters:

        encoded_tensor (torch.Tensor) : Tensor of encoded input.
        sep_token_id (int) : Token ID for [SEP] token.

    Returns:

        token_type_ids (torch.Tensor) : Tensor of token type IDs.
    """
    # Initialise tensor of zeroes
    token_type_ids = torch.zeros_like(encoded_tensor['input_ids']) 

    sep_positions = []
    for _, row in enumerate(encoded_tensor['input_ids'].tolist()):
        sep_list = []
        for j, token_id in enumerate(row):
            if token_id == sep_token_id:
                sep_list.append(j)
        sep_positions.append(sep_list)

    # After finding [SEP] tokens, must assign a unique token_type id
    for i, row in enumerate(sep_positions):
        for j, pos in enumerate(row): 
            # Final [SEP] token will be right before all the padding
            if j != len(row) - 1: 
                token_type_ids[i, pos + 1:row[j + 1] + 1] = j + 1 
        
    return token_type_ids

# --- Important functions for functionality of BERT ---

def tokenizer_encode(data, tokenizer : Union[BertTokenizer, AlbertTokenizer, RobertaTokenizer], truncation):
    """
    Helper function to encode textual data into a dictionary for use in training and testing.

    Parameters:
        
        data (pd.DataFrame) : The dataset.
        tokenizer (Union[BertTokenizer, AlbertTokenizer, RobertaTokenizer]) : Tokenizer, specific to the model used. 
        truncation (Union[None, str]) : The type of truncation to use. Valid options are None, 'head-only', 'tail-only' and 'head-and-tail'.

    Returns:

        encoded (dict) : Dictionary of tensors holding all necessary tensors for training and testing.
    """
    encoded = tokenizer.__call__(data, 
                                          add_special_tokens = True, 
                                          return_token_type_ids=True,
                                          padding = True, 
                                          truncation = truncation, 
                                          return_tensors = 'pt'
                                          )
    return encoded

def create_dataset(encoded_data, labels, categorical_features=None, numerical_features=None):
    """
    Helper function to create datasets from dictionaries obtained from tokenizer_encode().
    
    Parameters:
        
        encoded_data (dict) : The dictionary of tokens to put into a TensorDataset object.
        labels (torch.Tensor) : Target labels. 
        categorical_features (Union[None, list]) : Indicates the presence of categorical features to be added to the dataset.
        numerical_features (Union[None, list]) : Indicates the presence of numerical features to be added to the dataset.

    Returns:

        dataset (TensorDataset) : A TensorDataset object used to feed into the dataloader.

    """
    if categorical_features is not None and numerical_features is not None:
        dataset = TensorDataset(encoded_data['input_ids'], 
                                encoded_data['attention_mask'], 
                                encoded_data['token_type_ids'],
                                categorical_features, 
                                numerical_features,
                                labels
                                )

    elif categorical_features is not None:
        dataset = TensorDataset(encoded_data['input_ids'], 
                                encoded_data['attention_mask'], 
                                encoded_data['token_type_ids'],
                                categorical_features, 
                                labels
                                )
    elif numerical_features is not None:
        dataset = TensorDataset(encoded_data['input_ids'], 
                                encoded_data['attention_mask'], 
                                encoded_data['token_type_ids'],
                                numerical_features, 
                                labels
                                )
    else:
        dataset = TensorDataset(encoded_data['input_ids'], 
                                encoded_data['attention_mask'], 
                                encoded_data['token_type_ids'],
                                labels
                                )
    
    return dataset

def create_dataloader(dataset, batch_size: int, shuffle: bool=False):
    """
    Helper function to create dataloader from datasets.
    
    Parameters:
        
        dataset (TensorDataset) :  The dataset to put into a dataloader.
        batch_size (int) : The batch size used for each split.
        shuffle (bool) : Indicates whether to shuffle or keep the order of data. 

    Returns:

        dataloader (DataLoader) : A DataLoader object used for training and testing.
        
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def train_model(model, 
                model_name, 
                optimizer, 
                train_loader, 
                criterion, 
                epoch, 
                num_epochs, 
                categorical=False, 
                numerical=False, 
                scheduler=None):
    """
    Helper function to train the model.
    This is done for a single epoch.
    
    Parameters:
        
        model (Union[BertForSequenceClassification, BertWithTabular, 
        CustomBertForSequenceClassification, AlbertForSequenceClassification, 
        RobertaForSequenceClassification]) : The model to train.

        model_name (str) : Name of the model.
        optimizer (Union[Adam, AdamW]) : The optimizer used for optimising 
        model parameters.

        train_loader (DataLoader) : The dataloader used to load data in for 
        training.
        
        criterion (nn.CrossEntropyLoss) : The loss function to be minimised. 
        epoch (int) : The current training epoch.
        num_epochs (int) : The total number of training epochs.
        categorical (bool) : Indicates whether categorical features are used.
        numerical (bool) : Indicates whether numerical features are used.
        scheduler (Union[None, StepLR]) : The scheduler used for training.

    Returns:

        total_loss (float) : Total training loss.
        accuracy (float) : Training accuracy.
        
    """
    # Set model into training mode
    model.train()
    softmax = torch.nn.Softmax(dim=0)
    total_loss = 0.0
    correct = 0
    total = 0

    if model_name == 'custom' and categorical and numerical:
        for batch in tqdm(train_loader, 
                          desc=f'Epoch {epoch + 1}/{num_epochs}'):

            # Split to fit into width of dissertation
            input_batch = batch[0]
            attention_batch = batch[1]
            token_type_batch = batch[2]
            cat_batch = batch[3]
            num_batch = batch[4]
            target_batch = batch[5]

            # Zero gradients
            optimizer.zero_grad()
            
            # Feed parameters into model
            outputs = model(input_batch, 
                            attention_mask=attention_batch, 
                            token_type_ids=token_type_batch,
                            extra_data=torch.cat((cat_batch, num_batch), 
                                                 dim=1), 
                            labels = target_batch
                            )
            
            # Get loss from model output
            loss = outputs.loss

            # Backpropagation
            loss.backward()

            # Update gradients
            optimizer.step()

            # Collate loss
            total_loss += loss.item()
            logits = outputs.logits
            probs = softmax(logits)
            
            # Get most probable outputs
            preds = torch.argmax(probs, dim=1)
            total += target_batch.size(0)
            correct += (preds == target_batch).sum().item()
            
            if scheduler:
                scheduler.step()
    elif model_name == 'custom' and categorical:
        for input_batch, attention_batch, token_type_batch, cat_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(input_batch, 
                            attention_mask=attention_batch, 
                            token_type_ids=token_type_batch,
                            extra_data=cat_batch, 
                            labels = target_batch
                            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            logits = outputs.logits
            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)
            total += target_batch.size(0)
            correct += (preds == target_batch).sum().item()
            
            if scheduler:
                scheduler.step()

                    
    elif model_name == 'modular':
        # --Training--
        if categorical and numerical:
            for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                optimizer.zero_grad()
                loss, logits, _ = model(input_batch, 
                                attention_mask=attention_batch, 
                                token_type_ids=token_type_batch,
                                cat_feats=cat_batch, 
                                numerical_feats=num_batch, 
                                labels = target_batch
                                )
                probs = softmax(logits)
                preds = torch.argmax(probs, dim=1)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += target_batch.size(0)
                correct += (preds == target_batch).sum().item()

                if scheduler:
                    scheduler.step()

        elif categorical:
                for input_batch, attention_batch, token_type_batch, cat_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                    optimizer.zero_grad()
                    loss, logits, _ = model(input_batch, 
                                    attention_mask=attention_batch, 
                                    token_type_ids=token_type_batch,
                                    cat_feats=cat_batch, 
                                    labels = target_batch
                                    )
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=1)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += target_batch.size(0)
                    correct += (preds == target_batch).sum().item()

                    if scheduler:
                        scheduler.step()

        elif numerical:
                for input_batch, attention_batch, token_type_batch, num_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                    optimizer.zero_grad()
                    loss, logits, _ = model(input_batch, 
                                    attention_mask=attention_batch, 
                                    token_type_ids=token_type_batch,
                                    numerical_feats=num_batch, 
                                    labels = target_batch
                                    )
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=1)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += target_batch.size(0)
                    correct += (preds == target_batch).sum().item()
                    
                    if scheduler:
                        scheduler.step()
    else:
            for input_batch, attention_batch, token_type_batch, target_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                optimizer.zero_grad()
                outputs = model(input_batch, 
                                attention_mask = attention_batch, 
                                token_type_ids=token_type_batch,
                                labels = target_batch
                                )
                logits = outputs.logits
                probs = softmax(logits)
                preds = torch.argmax(probs, dim=1)
                loss = criterion(logits, target_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += target_batch.size(0)
                correct += (preds == target_batch).sum().item()

                if scheduler:
                    scheduler.step()
   
    accuracy = correct / total
    return total_loss, accuracy

def val_model(model, model_name, val_loader, criterion, epoch, num_epochs, categorical=False, numerical=False):
    """
    Helper function to validate the model.
    This is done for a single epoch.
    
    Parameters:
        
        model (Union[BertForSequenceClassification, BertWithTabular, CustomBertForSequenceClassification, AlbertForSequenceClassification, RobertaForSequenceClassification]) : The model to validate.
        model_name (str) : Name of the model.
        val_loader (DataLoader) : The dataloader used to load data in for validation.
        criterion (nn.CrossEntropyLoss) : The loss function to be minimised. 
        epoch (int) : The current training epoch.
        num_epochs (int) : The total number of training epochs.
        categorical (bool) : Indicates whether categorical features are used.
        numerical (bool) : Indicates whether numerical features are used.

    Returns:

        total_loss (float) : Total validation loss.
        accuracy (float) : Validation accuracy.
        torch.cat(all_preds) (torch.Tensor) : All predictions made.
        
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=0)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []

    if model_name == 'custom' and categorical and numerical:

        with torch.no_grad():
            for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                outputs = model(input_batch, 
                                attention_mask=attention_batch, 
                                token_type_ids=token_type_batch,
                                extra_data=torch.cat((cat_batch, num_batch), dim=1)
                                )
                logits = outputs.logits
                probs = softmax(logits)
                preds = torch.argmax(probs, dim=1)
                loss = criterion(logits, target_batch)
                total_loss += loss.item()
                total += target_batch.size(0)
                correct += (preds == target_batch).sum().item()
                all_preds.append(preds)
        
    elif model_name == 'custom' and categorical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                    outputs = model(input_batch, 
                                    attention_mask=attention_batch, 
                                    token_type_ids=token_type_batch,
                                    extra_data=cat_batch
                                    )
                    logits = outputs.logits
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=1)
                    loss = criterion(logits, target_batch)
                    total_loss += loss.item()
                    total += target_batch.size(0)
                    correct += (preds == target_batch).sum().item()
                    all_preds.append(preds)

                    
    elif model_name == 'modular':
        # --Training--
        if categorical and numerical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                    loss, logits, _ = model(input_batch, 
                                    attention_mask=attention_batch, 
                                    token_type_ids=token_type_batch,
                                    cat_feats=cat_batch, 
                                    numerical_feats=num_batch
                                    )
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=1)
                    loss = criterion(logits, target_batch)
                    total_loss += loss.item()
                    total += target_batch.size(0)
                    correct += (preds == target_batch).sum().item()
                    all_preds.append(preds)

        elif categorical:
                with torch.no_grad():
                    for input_batch, attention_batch, token_type_batch, cat_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                        loss, logits, _ = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        cat_feats=cat_batch
                                        )
                        probs = softmax(logits)
                        preds = torch.argmax(probs, dim=1)
                        loss = criterion(logits, target_batch)
                        total_loss += loss.item()
                        total += target_batch.size(0)
                        correct += (preds == target_batch).sum().item()
                        all_preds.append(preds)

        elif numerical:
                with torch.no_grad():
                    for input_batch, attention_batch, token_type_batch, num_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                        loss, logits, _ = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        numerical_feats=num_batch
                                        )
                        probs = softmax(logits)
                        preds = torch.argmax(probs, dim=1)
                        loss = criterion(logits, target_batch)
                        total_loss += loss.item()
                        total += target_batch.size(0)
                        correct += (preds == target_batch).sum().item() 
                        all_preds.append(preds)   
                    
    else:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, target_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                    outputs = model(input_batch, 
                                    attention_mask = attention_batch, 
                                    token_type_ids=token_type_batch
                                    )
                    logits = outputs.logits
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=1)
                    loss = criterion(logits, target_batch)
                    total_loss += loss.item()
                    total += target_batch.size(0)
                    correct += (preds == target_batch).sum().item()
                    all_preds.append(preds)
                
    accuracy = correct / total
    
    return total_loss, accuracy, torch.cat(all_preds)

def test_model(model, model_name, test_loader, categorical=False, numerical=False):
    """
    Helper function to test the model.
    This is done for a single epoch.
    
    Parameters:
        
        model (Union[BertForSequenceClassification, BertWithTabular, 
        CustomBertForSequenceClassification, AlbertForSequenceClassification, 
        RobertaForSequenceClassification]) : The model to test.

        model_name (str) : Name of the model.
        test_loader (DataLoader) : The dataloader used to load data in for testing.
        categorical (bool) : Indicates whether categorical features are used.
        numerical (bool) : Indicates whether numerical features are used.

    Returns:

        all_preds (torch.Tensor) : All predictions made.
        all_probs (torch.Tensor) : All probabilities for each class.
        
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=0)
    all_preds = []
    all_probs = []

    if model_name == 'custom' and categorical and numerical:
        with torch.no_grad():
            for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, _ in tqdm(test_loader, desc='Testing'):
                outputs  = model(input_batch, 
                                    attention_mask=attention_batch, 
                                    token_type_ids=token_type_batch,
                                    extra_data=torch.cat((cat_batch, num_batch), dim=1)
                                    )
                logits = outputs.logits
                probs = softmax(logits)
                all_probs.append(probs)
                preds = torch.argmax(probs, dim=1)
                for pred in preds.tolist():
                    all_preds.append(pred)
    elif model_name == 'custom' and categorical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, _ in tqdm(test_loader, desc='Testing'):
                    outputs  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        extra_data=cat_batch
                                        )
                    logits = outputs.logits
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)

    elif model_name == 'modular':
        if categorical and numerical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, _ in tqdm(test_loader, desc='Testing'):

                    _, logits, _  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        cat_feats=cat_batch, 
                                        numerical_feats=num_batch
                                        )
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)

        elif categorical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, _ in tqdm(test_loader, desc='Testing'):

                    _, logits, _  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        cat_feats=cat_batch
                                        )
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)

        elif numerical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, num_batch, _ in tqdm(test_loader, desc='Testing'):

                    _, logits, _  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        numerical_feats=num_batch
                                        )
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)
    
    else:
        with torch.no_grad():
            for input_batch, attention_batch, token_type_batch, _ in tqdm(test_loader, desc='Testing'):

                outputs = model(input_batch, 
                                attention_mask=attention_batch,
                                token_type_ids=token_type_batch
                                )
                logits = outputs.logits
                probs = softmax(logits)
                all_probs.append(probs)
                preds = torch.argmax(probs, dim=1)
                for pred in preds.tolist():
                    all_preds.append(pred)
    
    return all_preds, all_probs

def test_model_predict(model, model_name, test_loader, categorical=False, numerical=False):
    """
    Helper function to test the model for Phase 2.
    This is done for a single epoch.
    
    Parameters:
        
        model (Union[BertForSequenceClassification, BertWithTabular, CustomBertForSequenceClassification, AlbertForSequenceClassification, RobertaForSequenceClassification]) : The model to test.
        model_name (str) : Name of the model.
        test_loader (DataLoader) : The dataloader used to load data in for testing.
        categorical (bool) : Indicates whether categorical features are used.
        numerical (bool) : Indicates whether numerical features are used.

    Returns:

        all_preds (torch.Tensor) : All predictions made.
        all_probs (torch.Tensor) : All probabilities for each class.
        
    """
    model.eval()
    all_preds = []
    all_probs = []
    softmax = torch.nn.Softmax(dim=0)

    if model_name == 'custom':
        if categorical and numerical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, _ in test_loader:
                    outputs  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        extra_data=torch.cat((cat_batch, num_batch), dim=1)
                                        )
                    logits = outputs.logits
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)
        elif categorical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, _ in test_loader:
                    outputs  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        extra_data=cat_batch
                                        )
                    logits = outputs.logits
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)

    elif model_name == 'modular':
        if categorical and numerical:
            with torch.no_grad():
                for input_batch, attention_batch, token_type_batch, cat_batch, num_batch, _ in test_loader:
                    _, logits, _  = model(input_batch, 
                                        attention_mask=attention_batch, 
                                        token_type_ids=token_type_batch,
                                        cat_feats=cat_batch, 
                                        numerical_feats=num_batch
                                        )
                    probs = softmax(logits)
                    all_probs.append(probs)
                    preds = torch.argmax(probs, dim=1)
                    for pred in preds.tolist():
                        all_preds.append(pred)
    
    else:
        with torch.no_grad():
            for input_batch, attention_batch, token_type_batch, _ in test_loader:

                outputs = model(input_batch, 
                                attention_mask=attention_batch,
                                token_type_ids=token_type_batch
                                )
                logits = outputs.logits
                probs = softmax(logits)
                all_probs.append(probs)
                preds = torch.argmax(probs, dim=1)
                for pred in preds.tolist():
                    all_preds.append(pred)
    
    return all_preds, all_probs