import pandas as pd
from bert_models import *
from visualise import *

def main():

    data = pd.read_csv('../data/cleaned_and_synthesised.csv')
    data.dropna(inplace = True)

    # ---- Feature Representations ----

    print('CID')
    new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised'], model_name='custom', truncation_type='head-and-tail')

    # print('DoW')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'synthesised_dow_binarised'], model_name='custom')

    # print('Emojis and Emoticons')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'emojis', 'emoticons'], model_name='custom')

    # print('CID, DoW')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised', 'synthesised_dow_binarised'], model_name='custom')

    # print('CID, DoW, Emojis')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised', 'synthesised_dow_binarised', 'emojis', 'emoticons'], model_name='custom')

    # print('CID, DoW, POS (All)')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','ADP','PROPN','PRON','PART','INTJ','DET','X','SCONJ','CCONJ','PUNCT','SYM'], model_name='custom')

    # print('CID, DoW, POS (Main)')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ','ADV','PROPN', 'CCONJ', 'INTJ'], model_name='custom')

    # # ---- Truncation Types ----
    # print('Head Only')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised'], model_name='custom', truncation_type='head-only')

    # print('Tail Only')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised'], model_name='custom', truncation_type='tail-only')

    # print('Head and Tail')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised'], model_name='custom', truncation_type='head-and-tail')

    # print('Default')
    # new_bert(data, learning_rate=5e-5, batch_size=16, num_epochs=3, feature_cols=['content', 'binarised'], model_name='custom', truncation_type=None)

    # # ---- Hyperparameter Grid Search ----
    # for lr in [1e-5, 5e-5, 1e-4]:
    #     for bs in [8, 16, 32]:
    #         for e in [2, 3, 4, 5]:
    #             print('Learning Rate', lr)
    #             print('Batch Size', bs)
    #             print('Epochs', e)
    #             new_bert(data, learning_rate=lr, batch_size=bs, num_epochs=e, feature_cols=['content', 'binarised'], model_name='custom', truncation_type=None)

if __name__ == "__main__":
    main()