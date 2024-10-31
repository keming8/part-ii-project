import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gradboost_models import *
from testing_framework import *
from svm_models import *
from visualise import *

def main():
    data = pd.read_csv('../data/cleaned_and_synthesised.csv')
    

    # --- Choosing Different Vectorisers ---

    print("Count Vectorizer: ")
    cv_a, cv_p, cv_r, cv_f, cv_cm = new_gradboost(data, vectoriser = CountVectorizer(), time_series_split=True)

    # Plot HeatMap for Count Vectoriser
    plot_heatmap('vectoriser_count', data['thread_id'].unique().tolist(), cv_cm)
    
    # TFIDF
    print("TFIDF: ")
    tfidf_a, tfidf_p, tfidf_r, tfidf_f, tfidf_cm = new_gradboost(data, vectoriser = TfidfVectorizer(), time_series_split = True)

    # Plot HeatMap for TFIDF Vectoriser
    plot_heatmap('vectoriser_tfidf', data['thread_id'].unique().tolist(), tfidf_cm)

    # Bigrams
    print("Bigrams: ")
    bi_a, bi_p, bi_r, bi_f, bi_cm = new_gradboost(data, vectoriser = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b'))
    
    # Plot HeatMap for Bigram Vectoriser
    plot_heatmap('vectoriser_bigram', data['thread_id'].unique().tolist(), bi_cm)

    # Plot bar graph for vectorisers
    # This determines the vectoriser to use for the models
    plot_barplot('vectorisers', ['Count Vectoriser', 'TFIDF', 'Bigrams'], [[cv_a, tfidf_a, bi_a], [cv_p, tfidf_p, bi_p], [cv_r, tfidf_r, bi_r], [cv_f, tfidf_f, bi_f]])

    # --- Class Imbalances ---

    # Raw Data (Unstratified)
    print('Raw Data: ')
    #X_train_unstrat, X_test_unstrat, y_train_unstrat, y_test_unstrat = train_test_split(content, thread_ids, test_size = 0.3, random_state=100)
    #pre_train_tests(X_train_unstrat, X_test_unstrat, y_train_unstrat, y_test_unstrat)
    raw_a, raw_p, raw_r, raw_f, raw_cm = new_gradboost(data, stratify=False)

    # Plot HeatMap for unstratified data
    plot_heatmap('imbalance_unstratified', data['thread_id'].unique().tolist(), raw_cm)

    # Stratified Data
    # Use TFIDF results from above

    # Oversampled using SMOTE
    print('SMOTE: ')
    smote_a, smote_p, smote_r, smote_f, smote_cm = new_gradboost(data, oversample = 'smote')

    # Plot HeatMap for SMOTE oversampling
    plot_heatmap('imbalance_smote', data['thread_id'].unique().tolist(), smote_cm)

    # Oversampled using Duplication
    print('Duplication: ')
    dupe_a, dupe_p, dupe_r, dupe_f, dupe_cm = new_gradboost(data, oversample = 'duplicate')

    # Plot HeatMap for duplicated oversampling
    plot_heatmap('imbalance_duplicate', data['thread_id'].unique().tolist(), dupe_cm)

    # Plot bar graph for class imbalances
    plot_barplot('Handling Class Imbalance', ['Raw', 'Stratified', 'Duplicated', 'SMOTE'], [[raw_a, tfidf_a, dupe_a, smote_a], [raw_p, tfidf_p, dupe_p, smote_p], [raw_r, tfidf_r, dupe_r, smote_r], [raw_f, tfidf_f, dupe_f, smote_f]])

    # --- Message Time Representations ---
    # Using Timestamps in feature vectors
    print('Timestamps: ')
    timestamp_a , timestamp_p, timestamp_r, timestamp_f, timestamp_cm = new_gradboost(data, feature_cols=['tokens', 'timestamps'])

    # Plot HeatMap for Timestamp Time representation
    plot_heatmap('time_timestamp', data['thread_id'].unique().tolist(), timestamp_cm)

    # Using time offsets in feature vectors
    print('Offsets: ')
    offset_a , offset_p, offset_r, offset_f, offset_cm = new_gradboost(data, feature_cols=['tokens', 'offsets'])

    # Plot HeatMap for Offset Time representation
    plot_heatmap('time_offset', data['thread_id'].unique().tolist(), offset_cm)

    # Days of Week (Str)
    print('Day of Week (int): ')
    dows_a, dows_p, dows_r, dows_f, dows_cm = new_gradboost(data, feature_cols=['tokens', 'synthesised_day_of_week_int'])

    # Plot HeatMap for Day of Week (Str) Time representation
    plot_heatmap('time_day_of_week_int', data['thread_id'].unique().tolist(), dows_cm)

    # Date
    print('Date: ')
    date_a, date_p, date_r, date_f, date_cm = new_gradboost(data, feature_cols=['tokens', 'synthesised_days', 'synthesised_months', 'synthesised_years'])
    # Plot HeatMap for Offset Time representation
    plot_heatmap('time_date_str', data['thread_id'].unique().tolist(), date_cm)

    # Plot bar graph for message time representations
    plot_barplot('time_representations', ['Timestamps', 'Offsets', 'DOW(int)', 'Date'], [[timestamp_a, offset_a, dows_a, date_a], [timestamp_p, offset_p, dows_p, date_p], [timestamp_r, offset_r, dows_r, date_r], [timestamp_f, offset_f, dows_f, date_f]])


    # --- CID Representations ---
    # Using cids (Padded lists)
    print('CIDs Padded List: ')
    padlist_a, padlist_p, padlist_r, padlist_f, padlist_cm = new_gradboost(data, feature_cols=['tokens', 'all_cids'])

    # Plot HeatMap for Padded List cid representation
    plot_heatmap('cid_padlist', data['thread_id'].unique().tolist(), padlist_cm)

    # Using CIDs (Binarised)
    print('CIDs Binarised: ')
    binarised_a, binarised_p, binarised_r, binarised_f, binarised_cm = new_gradboost(data, feature_cols=['tokens', 'binarised'])

    # Plot HeatMap for Binarised CID representation
    plot_heatmap('cid_binarised', data['thread_id'].unique().tolist(), binarised_cm)

    # Plot bar graph for CID representations
    plot_barplot('cid_representations', ['Padded List', 'Binarised'], [[padlist_a, binarised_a], [padlist_p, binarised_p], [padlist_r, binarised_r], [padlist_f, binarised_f]])
    
    # --- Combining Features ---
    # Use both time representations and cid representations

    # Timestamps and Padded List
    print('Timestamps and Padded List: ')
    tp_a, tp_p, tp_r, tp_f, tp_cm = new_gradboost(data, feature_cols=['tokens', 'timestamps', 'all_cids'])
    #plot_heatmap('combined_timestamp_padlist', data['thread_id'].unique().tolist(), tp_cm)

    # Timestamps and Binarised
    print('Timestamps and Binarised: ')
    tb_a, tb_p, tb_r, tb_f, tb_cm = new_gradboost(data, feature_cols=['tokens', 'timestamps', 'binarised'])
    #plot_heatmap('combined_timestamp_binarised', data['thread_id'].unique().tolist(), tb_cm)

    # Offsets and Padded List
    print('Offsets and Padded List: ')
    op_a, op_p, op_r, op_f, op_cm = new_gradboost(data, feature_cols=['tokens', 'offsets', 'all_cids'])
    #plot_heatmap('combined_offset_padlist', data['thread_id'].unique().tolist(), op_cm)
    
    # Offsets and Binarised
    print('Offsets and Binarised: ')
    ob_a, ob_p, ob_r, ob_f, ob_cm = new_gradboost(data, feature_cols=['tokens', 'offsets', 'binarised'])
    #plot_heatmap('combined_offset_binarised', data['thread_id'].unique().tolist(), ob_cm)

    # Plot bar graph for all combinations, as well as representations for message time and cids
    plot_barplot('combinations', ['Timestamps', 'Offsets', 'Padded List CIDs', 'Binarised CIDs', 'Timestamp - Padded List', 'Timestamp - Binarised', 'Offsets - Padded List', 'Offset - Binarised'], [[timestamp_a, offset_a, padlist_a, binarised_a, tp_a, tb_a, op_a, ob_a], [timestamp_p, offset_p, padlist_p, binarised_p, tp_p, tb_p, op_p, ob_p], [timestamp_r, offset_r, padlist_r, binarised_r, tp_r, tb_r, op_r, ob_r], [timestamp_f, offset_f, padlist_f, binarised_f, tp_f, tb_f, op_f, ob_f]])

    # -- Time Series Cross Validation --
    print('Normal Train Test Split: ')
    normal_a, normal_p, normal_r, normal_f, normal_cm = new_gradboost(data)
    #plot_heatmap('tts_normal', data['thread_id'].unique().tolist(), normal_cm)

    print('Time Series Train Test Split: ')
    ts_a, ts_p, ts_r, ts_f, ts_cm = new_gradboost(data, time_series_split=True, time_series_split_n=3)

    #plot_heatmap('tts_time_series', data['thread_id'].unique().tolist(), ts_cm)
    print('Time Series Train Test Split (n = 3): ')
    ts_a3, ts_p3, ts_r3, ts_f3, ts_cm = new_gradboost(data, time_series_split=True, time_series_split_n=2)
    print('Time Series Train Test Split (n = 4): ')
    ts_a4, ts_p4, ts_r4, ts_f4, ts_cm = new_gradboost(data, time_series_split=True, time_series_split_n=3)
    print('Time Series Train Test Split (n = 8): ')
    ts_a5, ts_p5, ts_r5, ts_f5, ts_cm = new_gradboost(data, time_series_split=True, time_series_split_n=8)
    print('Time Series Train Test Split (n = 10): ')
    ts_a6, ts_p6, ts_r6, ts_f6, ts_cm = new_gradboost(data, time_series_split=True, time_series_split_n=10)
    

    plot_barplot('time_series_n_2', ['3', '4', '5', '6'], [[ts_a3, ts_a4, ts_a5, ts_a6], [ts_p3, ts_p4, ts_p5, ts_p6], [ts_r3, ts_r4, ts_r5, ts_r6], [ts_f3, ts_f4, ts_f5, ts_f6]])

    # Plot bar graph for different train test splits
    plot_barplot('train_test_splits', ['Normal', 'Time Series'], [[normal_a, ts_a], [normal_p, ts_p], [normal_r, ts_r], [normal_f, ts_f]])
    
    # --- Few Shot Learning ---
    print('Few Shot 10%: ')
    fs_10_a, fs_10_p, fs_10_r, fs_10_f, fs_10_cm = new_gradboost(data, few_shot=0.1)
    plot_heatmap('few_shot_10', data['thread_id'].unique().tolist(), fs_10_cm)
    
    print('Few Shot 20%: ')
    fs_20_a, fs_20_p, fs_20_r, fs_20_f, fs_20_cm = new_gradboost(data, few_shot=0.2)
    plot_heatmap('few_shot_20', data['thread_id'].unique().tolist(), fs_20_cm)

    print('Few Shot 50%: ')
    fs_50_a, fs_50_p, fs_50_r, fs_50_f, fs_50_cm = new_gradboost(data, few_shot=0.5)
    plot_heatmap('few_shot_50', data['thread_id'].unique().tolist(), fs_50_cm)

    # Plot bar graph for different tests
    plot_barplot('few_shot_learning', ['10%', '20%', '50%'], [[fs_10_a, fs_20_a, fs_50_a], [fs_10_p, fs_20_p, fs_50_p], [fs_10_r, fs_20_r, fs_50_r], [fs_10_f, fs_20_f, fs_50_f]])
    
    

    # --- Adding Noise and Perturbing Data ---
    # augmentX can be any of ['synonym', 'word_deletion', 'swap', 'spelling', 'char_replacement']
    print('Perturb X Synonym: ')
    x_syn_a, x_syn_p, x_syn_r, x_syn_f, x_syn_cm = new_gradboost(data, augmentX='synonym')
    plot_heatmap('perturb_X_syn', data['thread_id'].unique().tolist(), x_syn_cm)

    print('Perturb X Word Deletion: ')
    x_del_a, x_del_p, x_del_r, x_del_f, x_del_cm = new_gradboost(data, augmentX='word_deletion')
    plot_heatmap('perturb_X_del', data['thread_id'].unique().tolist(), x_del_cm)

    print('Perturb X Swap: ')
    x_swap_a, x_swap_p, x_swap_r, x_swap_f, x_swap_cm = new_gradboost(data, augmentX='swap')
    plot_heatmap('perturb_X_swap', data['thread_id'].unique().tolist(), x_swap_cm)

    print('Perturb X Spelling: ')
    x_spell_a, x_spell_p, x_spell_r, x_spell_f, x_spell_cm = new_gradboost(data, augmentX='spelling')
    plot_heatmap('perturb_X_spell', data['thread_id'].unique().tolist(), x_spell_cm)

    print('Perturb X Char Replacement: ')
    x_char_a, x_char_p, x_char_r, x_char_f, x_char_cm = new_gradboost(data, augmentX='char_replacement')
    plot_heatmap('perturb_X_char', data['thread_id'].unique().tolist(), x_char_cm)

    print('Perturb Y: ')
    y_label_a, y_label_p, y_label_r, y_label_f, y_label_cm = new_gradboost(data, augmentY=True)
    plot_heatmap('perturb_X_char', data['thread_id'].unique().tolist(), y_label_cm)

    # Plot bar graph for tests on perturbed training data
    plot_barplot('training_noise', ['X_Syn', 'X_Del', 'X_Swap', 'X_Spell', 'X_Char', 'Y'], [[x_syn_a, x_del_a, x_swap_a, x_spell_a, x_char_a, y_label_a], [x_syn_p, x_del_p, x_swap_p, x_spell_p, x_char_p, y_label_p], [x_syn_r, x_del_r, x_swap_r, x_spell_r, x_char_r, y_label_r], [x_syn_f, x_del_f, x_swap_f, x_spell_f, x_char_f, y_label_f]])
    

    # --- Use POS Tags ---
    print('POS Tags All: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_gradboost(data, feature_cols=['tokens','NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','ADP','PROPN','PRON','PART','INTJ','DET','X','SCONJ','CCONJ','PUNCT','SYM'])
    print('POS Tags Some: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_gradboost(data, feature_cols=['tokens', 'NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','PROPN'])
    print('POS Tags Main: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_gradboost(data, feature_cols=['tokens', 'NOUN','VERB', 'ADJ'])
    print('POS Tags Main 2: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_gradboost(data, feature_cols=['tokens', 'NOUN','VERB', 'ADJ', 'ADV','PROPN'])
    print('POS Tags Main 3: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_gradboost(data, feature_cols=['tokens', 'NOUN','VERB', 'ADJ', 'ADV'])


    

    # --- Testing on larger Dataset ---

    # -- Normal Train Test Split, TFIDF, SMOTE, Sigmoid kernel, CIDs binarised
    test1_a, test1_p, test1_r, test1_f, test1_cm = new_gradboost(data, feature_cols=['tokens', 'binarised'], vectoriser= TfidfVectorizer(), oversample='smote')
    plot_heatmap('large_test_1', data['thread_id'].unique().tolist(), test1_cm)
    
    test2_a, test2_p, test2_r, test2_f, test2_cm = new_gradboost(data, feature_cols=['tokens', 'binarised'], vectoriser= TfidfVectorizer())

    plot_heatmap('large_test_2', data['thread_id'].unique().tolist(), test2_cm)
    
    test3_a, test3_p, test3_r, test3_f, test3_cm = new_gradboost(data, feature_cols=['tokens', 'binarised'], vectoriser= TfidfVectorizer(), time_series_split=True, time_series_split_n=4)
    plot_heatmap('large_test_3', data['thread_id'].unique().tolist(), test3_cm)

    # Plot bar graph for different tests
    plot_barplot('large_tests', ['Test1', 'Test2', 'Test3'], [[test1_a, test2_a, test3_a], [test1_p, test2_p, test3_p], [test1_r, test2_r, test3_r], [test1_f, test2_f, test3_f]])
    
    # --- Hyperparameter Grid Search ---
    hps1_a, hps1_p, hps1_r, hps1_f, hps1_cm = new_gradboost(data, n_estimators=100, learning_rate=0.01, max_depth=3)
    hps2_a, hps2_p, hps2_r, hps2_f, hps2_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=3)
    hps3_a, hps3_p, hps3_r, hps3_f, hps3_cm = new_gradboost(data, n_estimators=200, learning_rate=0.01, max_depth=3)
    hps4_a, hps4_p, hps4_r, hps4_f, hps4_cm = new_gradboost(data, n_estimators=100, learning_rate=0.05, max_depth=3)
    hps5_a, hps5_p, hps5_r, hps5_f, hps5_cm = new_gradboost(data, n_estimators=150, learning_rate=0.05, max_depth=3)
    hps6_a, hps6_p, hps6_r, hps6_f, hps6_cm = new_gradboost(data, n_estimators=200, learning_rate=0.05, max_depth=3)
    hps7_a, hps7_p, hps7_r, hps7_f, hps7_cm = new_gradboost(data, n_estimators=100, learning_rate=0.1, max_depth=3)
    hps8_a, hps8_p, hps8_r, hps8_f, hps8_cm = new_gradboost(data, n_estimators=150, learning_rate=0.1, max_depth=3)
    hps9_a, hps9_p, hps9_r, hps9_f, hps9_cm = new_gradboost(data, n_estimators=200, learning_rate=0.1, max_depth=3)

    plot_barplot('gradboost_hyperparam_search', ['100_0.01', '150_0.01', '200_0.01', '100_0.05', '150_0.05', '200_0.05', '100_0.1', '150_0.1', '200_0.1'], 
                 [[hps1_a, hps2_a, hps3_a, hps4_a, hps5_a, hps6_a, hps7_a, hps8_a, hps9_a], 
                  [hps1_p, hps2_p, hps3_p, hps4_p, hps5_p, hps6_p, hps7_p, hps8_p, hps9_p], 
                  [hps1_r, hps2_r, hps3_r, hps4_r, hps5_r, hps6_r, hps7_r, hps8_r, hps9_r], 
                  [hps1_f, hps2_f, hps3_f, hps4_f, hps5_f, hps6_f, hps7_f, hps8_f, hps9_f]])
    
    hps1_a, hps1_p, hps1_r, hps1_f, hps1_cm = new_gradboost(data, n_estimators=150, learning_rate=0.001, max_depth=2)
    hps2_a, hps2_p, hps2_r, hps2_f, hps2_cm = new_gradboost(data, n_estimators=150, learning_rate=0.001, max_depth=3)
    hps3_a, hps3_p, hps3_r, hps3_f, hps3_cm = new_gradboost(data, n_estimators=150, learning_rate=0.001, max_depth=4)
    hps4_a, hps4_p, hps4_r, hps4_f, hps4_cm = new_gradboost(data, n_estimators=150, learning_rate=0.005, max_depth=2)
    hps5_a, hps5_p, hps5_r, hps5_f, hps5_cm = new_gradboost(data, n_estimators=150, learning_rate=0.005, max_depth=3)
    hps6_a, hps6_p, hps6_r, hps6_f, hps6_cm = new_gradboost(data, n_estimators=150, learning_rate=0.005, max_depth=4)
    hps7_a, hps7_p, hps7_r, hps7_f, hps7_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=2)
    hps8_a, hps8_p, hps8_r, hps8_f, hps8_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=3)

    print('Few Shot 10')
    hps9_a, hps9_p, hps9_r, hps9_f, hps9_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=4)
    print('Few Shot 20')
    hps9_a, hps9_p, hps9_r, hps9_f, hps9_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=4, few_shot=0.2)
    print('Few Shot 50')
    hps9_a, hps9_p, hps9_r, hps9_f, hps9_cm = new_gradboost(data, n_estimators=150, learning_rate=0.01, max_depth=4, few_shot=0.5)

    plot_barplot('gradboost_hyperparam_search_2', ['0.001_2', '0.001_3', '0.001_4', '0.005_2', '0.005_3', '0.005_4', '0.01_2', '0.01_3', '0.01_4'], 
                 [[hps1_a, hps2_a, hps3_a, hps4_a, hps5_a, hps6_a, hps7_a, hps8_a, hps9_a], 
                  [hps1_p, hps2_p, hps3_p, hps4_p, hps5_p, hps6_p, hps7_p, hps8_p, hps9_p], 
                  [hps1_r, hps2_r, hps3_r, hps4_r, hps5_r, hps6_r, hps7_r, hps8_r, hps9_r], 
                  [hps1_f, hps2_f, hps3_f, hps4_f, hps5_f, hps6_f, hps7_f, hps8_f, hps9_f]])
    
    
if __name__ == "__main__":
    main()