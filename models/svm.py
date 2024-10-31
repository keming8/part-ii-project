import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from svm_models import *
from visualise import *

def main():
    data = pd.read_csv('../data/cleaned_and_synthesised.csv')
    
    # --- Choosing Different Vectorisers ---

    # -- Count Vectoriser --
    print("Count Vectorizer: ")
    cv_a, cv_p, cv_r, cv_f, cv_cm = new_svm(data, vectoriser = CountVectorizer())

    # Plot HeatMap for Count Vectoriser
    plot_heatmap('vectoriser_count', data['thread_id'].unique().tolist(), cv_cm)

    # -- TFIDF -- 
    print("TFIDF: ")
    tfidf_a, tfidf_p, tfidf_r, tfidf_f, tfidf_cm = new_svm(data, vectoriser = TfidfVectorizer())
    
    # Plot HeatMap for TFIDF Vectoriser
    plot_heatmap('vectoriser_tfidf', data['thread_id'].unique().tolist(), tfidf_cm)

    # -- Bigrams -- 
    print("Bigrams: ")
    bi_a, bi_p, bi_r, bi_f, bi_cm = new_svm(data, vectoriser = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b'))
    
    # Plot HeatMap for Bigram Vectoriser
    plot_heatmap('vectoriser_bigram', data['thread_id'].unique().tolist(), bi_cm)

    # Plot bar graph for vectorisers
    plot_barplot('vectorisers', ['Count Vectoriser', 'TFIDF', 'Bigrams'], [[cv_a, tfidf_a, bi_a], [cv_p, tfidf_p, bi_p], [cv_r, tfidf_r, bi_r], [cv_f, tfidf_f, bi_f]])


    # --- Class Imbalances ---

    # -- Raw Data (Unstratified) --
    print('Raw Data: ')
    raw_a, raw_p, raw_r, raw_f, raw_cm = new_svm(data, stratify=False)

    # Plot HeatMap for unstratified data
    plot_heatmap('imbalance_unstratified', data['thread_id'].unique().tolist(), raw_cm)

    # -- Stratified Data --
    print('Stratified')
    strat_a, strat_p, strat_r, strat_f, strat_cm = new_svm(data, stratify=True)

    # Plot HeatMap for stratified data
    plot_heatmap('imbalance_stratified', data['thread_id'].unique().tolist(), strat_cm)

    # -- Oversampled using SMOTE --
    print('SMOTE: ')
    smote_a, smote_p, smote_r, smote_f, smote_cm = new_svm(data, oversample = 'smote')

    # Plot HeatMap for SMOTE oversampling
    plot_heatmap('imbalance_smote', data['thread_id'].unique().tolist(), smote_cm)

    # -- Oversampled using Duplication --
    print('Duplication: ')
    dupe_a, dupe_p, dupe_r, dupe_f, dupe_cm = new_svm(data, oversample = 'duplicate')

    # Plot HeatMap for duplicated oversampling
    plot_heatmap('imbalance_duplicate', data['thread_id'].unique().tolist(), dupe_cm)

    # Plot bar graph for class imbalances
    plot_barplot('Handling Class Imbalance', 
                 ['Raw', 'Stratified', 'Duplicated', 'SMOTE'], 
                 [[raw_a, strat_a, dupe_a, smote_a], 
                  [raw_p, strat_p, dupe_p, smote_p], 
                  [raw_r, strat_r, dupe_r, smote_r], 
                  [raw_f, strat_f, dupe_f, smote_f]])
    

    # --- Choosing Different Kernel Functions ---
    # linear, poly, rbf, sigmoid

    # -- Linear --
    print('Linear Kernel: ')
    linear_a, linear_p, linear_r, linear_f, linear_cm = new_svm(data, kernel = 'linear')

    # Plot HeatMap for Linear kernel function
    plot_heatmap('kernel_linear', data['thread_id'].unique().tolist(), linear_cm)

    # -- Polynomial --
    print('Polynomial Kernel: ')
    poly_a, poly_p, poly_r, poly_f, poly_cm = new_svm(data, kernel = 'poly')

    # Plot HeatMap for Polynomial kernel function
    plot_heatmap('kernel_polynomial', data['thread_id'].unique().tolist(), poly_cm)

    # -- Radial Basis Function --
    print('RBF Kernel: ')
    rbf_a, rbf_p, rbf_r, rbf_f, rbf_cm = new_svm(data, kernel = 'rbf')

    # Plot HeatMap for RBF kernel function
    plot_heatmap('kernel_rbf', data['thread_id'].unique().tolist(), rbf_cm)

    # -- Sigmoid --
    print('Sigmoid Kernel: ')
    sig_a, sig_p, sig_r, sig_f, sig_cm = new_svm(data, kernel = 'sigmoid')

    # Plot HeatMap for Sigmoid kernel function
    plot_heatmap('kernel_sigmoid', data['thread_id'].unique().tolist(), sig_cm)

    # Plot bar graph for kernel functions
    plot_barplot('kernel_functions', 
                 ['Linear', 'Polynomial', 'RBF', 'Sigmoid'], 
                 [[linear_a, poly_a, rbf_a, sig_a], 
                  [linear_p, poly_p, rbf_p, sig_p], 
                  [linear_r, poly_r, rbf_r, sig_r], 
                  [linear_f, poly_f, rbf_f, sig_f]])
    
    # --- OVO vs OVR ---
    
    # -- One vs One --
    print('One vs One: ')
    ovo_a, ovo_p, ovo_r, ovo_f, ovo_cm = new_svm(data, decision_function_shape='ovo')

    # Plot HeatMap for One vs One Decision Function
    plot_heatmap('decision_ovo', data['thread_id'].unique().tolist(), ovo_cm)

    # -- One vs Rest --
    print('One vs Rest: ')
    ovr_a, ovr_p, ovr_r, ovr_f, ovr_cm = new_svm(data, decision_function_shape='ovr')

    # Plot HeatMap for One vs Rest Decision Function
    plot_heatmap('decision_ovr', data['thread_id'].unique().tolist(), ovr_cm)

    # Plot bar graph for decision function shapes
    plot_barplot('decision_functions', ['OVO', 'OVR'], [[ovo_a, ovr_a], [ovo_p, ovr_p], [ovo_r, ovr_r], [ovo_f, ovr_f]])

    # ----- Trying Different Feature Representations -----

    # --- Message Time Representations ---

    # -- Timestamps --
    print('Timestamps: ')
    timestamp_a , timestamp_p, timestamp_r, timestamp_f, timestamp_cm = new_svm(data, ['tokens', 'timestamps'])

    # Plot HeatMap for Timestamp Time representation
    plot_heatmap('time_timestamp', data['thread_id'].unique().tolist(), timestamp_cm)

    # -- Offsets --
    print('Offsets: ')
    offset_a , offset_p, offset_r, offset_f, offset_cm = new_svm(data, ['tokens', 'offsets'])

    # Plot HeatMap for Offset Time representation
    plot_heatmap('time_offset', data['thread_id'].unique().tolist(), offset_cm)

    # -- Days of Week (Int) --
    print('Day of Week (int): ')
    dows_a, dows_p, dows_r, dows_f, dows_cm = new_svm(data, ['tokens', 'synthesised_day_of_week_int'])

    # Plot HeatMap for Day of Week (Str)
    plot_heatmap('time_day_of_week_int', data['thread_id'].unique().tolist(), dows_cm)

    # -- Days of Week (One-Hot Encoding) --
    print('Day of Week (binarised): ')
    dowb_a, dowb_p, dowb_r, dowb_f, dowb_cm = new_svm(data, ['tokens', 'synthesised_dow_binarised'])

    # Plot HeatMap for Day of Week (One-Hot Encoding)
    plot_heatmap('time_day_of_week_int', data['thread_id'].unique().tolist(), dowb_cm)

    # -- Date --
    print('Date: ')
    date_a, date_p, date_r, date_f, date_cm = new_svm(data, ['tokens', 'synthesised_days', 'synthesised_months', 'synthesised_years'])

    # Plot HeatMap for Date 
    plot_heatmap('time_date', data['thread_id'].unique().tolist(), date_cm)

    # -- One-Hot Encoded Date --
    print('One-Hot Encoded Date: ')
    bin_date_a, bin_date_p, bin_date_r, bin_date_f, bin_date_cm = new_svm(data, ['tokens', 'synthesised_day_binarised', 'synthesised_month_binarised', 'synthesised_year_binarised'])

    # Plot HeatMap for One-Hot Encoded Date 
    plot_heatmap('time_binarised_date', data['thread_id'].unique().tolist(), bin_date_cm)

    # Plot bar graph for message time representations
    plot_barplot('time_representations', 
                 ['Timestamps', 'Offsets', 'Date', 'Binarised Date'], 
                 [[timestamp_a, offset_a, date_a, bin_date_a], 
                  [timestamp_p, offset_p, date_p, bin_date_p], 
                  [timestamp_r, offset_r, date_r, bin_date_r], 
                  [timestamp_f, offset_f, date_f, bin_date_f]])
    plot_barplot('day_of_week_representations', ['Int', 'Binarised'], [[dows_a, dowb_a], [dows_p, dowb_p], [dows_r, dowb_r], [dows_f, dowb_f]])
    
    # --- CID Representations ---

    # -- Padded List --
    print('CIDs Padded List: ')
    padlist_a, padlist_p, padlist_r, padlist_f, padlist_cm = new_svm(data, ['tokens', 'all_cids'])

    # Plot HeatMap for Padded List cid representation
    plot_heatmap('cid_padlist', data['thread_id'].unique().tolist(), padlist_cm)

    # -- One-Hot Encoded --
    print('CIDs One-Hot Encoded: ')
    binarised_a, binarised_p, binarised_r, binarised_f, binarised_cm = new_svm(data, ['tokens', 'binarised'])
    
    # Plot HeatMap for One-Hot Encoded CID representation
    plot_heatmap('cid_binarised', data['thread_id'].unique().tolist(), binarised_cm)

    # Plot bar graph for CID representations
    plot_barplot('cid_representations', ['Padded List', 'Binarised'], [[padlist_a, binarised_a], [padlist_p, binarised_p], [padlist_r, binarised_r], [padlist_f, binarised_f]])
    

    # --- Emojis and Emoticons ---

    # -- Emojis Only --
    print('Emojis Only: ')
    emoji_a, emoji_p, emoji_r, emoji_f, emoji_cm = new_svm(data, ['tokens', 'emojis'])

    # Plot HeatMap for Emojis
    plot_heatmap('emoji_emoji', data['thread_id'].unique().tolist(), emoji_cm)

    # -- Emoticons Only --
    print('Emoticons Only: ')
    emoticon_a, emoticon_p, emoticon_r, emoticon_f, emoticon_cm = new_svm(data, ['tokens', 'emoticons'])

    # Plot HeatMap for Emoticons
    plot_heatmap('emoji_emoticons', data['thread_id'].unique().tolist(), emoticon_cm)
    
    # -- Emojis and Emoticons --
    print('Emoticons and Emojis Separate: ')
    sep_a, sep_p, sep_r, sep_f, sep_cm = new_svm(data, ['tokens', 'emoticons', 'emojis'])

    # Plot HeatMap for Emojis and Emoticons Separate
    plot_heatmap('emoji_separate', data['thread_id'].unique().tolist(), sep_cm)

    # -- Emojis or Emoticons --
    print('Emoticons and Emojis Together: ')
    or_a, or_p, or_r, or_f, or_cm = new_svm(data, ['tokens', 'emojis_and_emoticons'])

    # Plot HeatMap for Emojis and Emoticons Together
    plot_heatmap('emoji_separate', data['thread_id'].unique().tolist(), or_cm)

    # Plot bar graph for CID representations
    plot_barplot('emoji_representations', 
                 ['Emojis', 'Emoticons', 'Separate', 'Together'], 
                 [[emoji_a, emoticon_a, sep_a, or_a], 
                  [emoji_p, emoticon_p, sep_p, or_p], 
                  [emoji_r, emoticon_r, sep_r, or_r], 
                  [emoji_f, emoticon_f, sep_f, or_f]])
    
    # --- POS Tags ---
    print('POS Tags All: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_svm(data, ['tokens','NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','ADP','PROPN','PRON','PART','INTJ','DET','X','SCONJ','CCONJ','PUNCT','SYM'])
    print('POS Tags Some: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_svm(data, ['tokens', 'NOUN','VERB', 'ADJ','ADV','PROPN', 'CCONJ', 'INTJ'])
    print('POS Tags Main: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_svm(data, ['tokens', 'NOUN','VERB', 'ADJ'])
    print('POS Tags Main 2: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_svm(data, ['tokens', 'NOUN','VERB', 'ADJ', 'ADV','PROPN'])
    print('POS Tags Main 3: ')
    pos_a, pos_p, pos_r, pos_f, pos_cm = new_svm(data, ['tokens', 'NOUN','VERB', 'ADJ', 'ADV'])
    
    # ---- Combining Features ----
    
    # -- CID, Emojis --
    print('CIDs, Emojis: ')
    a1, p1, r1, f1, cm1 = new_svm(data, ['tokens', 'binarised', 'emojis_and_emoticons'])

    # Plot HeatMap for CID, Emojis 
    plot_heatmap('combined_1', data['thread_id'].unique().tolist(), cm1)

    # -- CID, Day of Week --
    print('CIDs, DoW: ')
    a2, p2, r2, f2, cm2 = new_svm(data, ['tokens', 'binarised', 'synthesised_dow_binarised'])

    # Plot HeatMap for CID, DoW
    plot_heatmap('combined_2', data['thread_id'].unique().tolist(), cm2)

    # -- CID, Day of Week --
    print('CIDs, DoW, Emojis: ')
    a3, p3, r3, f3, cm3 = new_svm(data, ['tokens', 'binarised', 'synthesised_dow_binarised', 'emojis_and_emoticons'])

    # Plot HeatMap for CID, DoW, Emojis
    plot_heatmap('combined_3', data['thread_id'].unique().tolist(), cm3)

    # -- CID, Day of Week, POS (All) --
    print('CIDs, DoW, POS (All): ')
    a4, p4, r4, f4, cm4 = new_svm(data, ['tokens', 'binarised', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','ADP','PROPN','PRON','PART','INTJ','DET','X','SCONJ','CCONJ','PUNCT','SYM'])

    # Plot HeatMap for CID, DoW, POS (All)
    plot_heatmap('combined_4', data['thread_id'].unique().tolist(), cm4)

    # -- CID, Day of Week, POS (Main) --
    print('CIDs, DoW, POS (Main): ')
    a5, p5, r5, f5, cm5 = new_svm(data, ['tokens', 'binarised', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ','ADV','PROPN', 'CCONJ', 'INTJ'])

    # Plot HeatMap for CID, DoW, POS (Main)
    plot_heatmap('combined_5', data['thread_id'].unique().tolist(), cm5)

    # -- CID, Day of Week, POS (All), Emojis --
    print('CIDs, DoW, POS (All): ')
    a6, p6, r6, f6, cm6 = new_svm(data, ['tokens', 'binarised', 'emojis_and_emoticons', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ', 'AUX','NUM', 'ADV','ADP','PROPN','PRON','PART','INTJ','DET','X','SCONJ','CCONJ','PUNCT','SYM'])

    # Plot HeatMap for CID, DoW, POS (All), Emojis
    plot_heatmap('combined_6', data['thread_id'].unique().tolist(), cm6)

    # -- CID, Day of Week, POS (Main), Emojis --
    print('CIDs, DoW, POS (Main): ')
    a7, p7, r7, f7, cm7 = new_svm(data, ['tokens', 'binarised', 'emojis_and_emoticons', 'synthesised_dow_binarised', 'NOUN','VERB', 'ADJ','ADV','PROPN', 'CCONJ', 'INTJ'])

    # Plot HeatMap for CID, DoW, POS (Main), Emojis
    plot_heatmap('combined_7', data['thread_id'].unique().tolist(), cm7)

    # Plot bar graph for CID representations
    plot_barplot('combined', 
                 [1, 2, 3, 4, 5, 6, 7], 
                 [[a1, a2, a3, a4, a5, a6, a7], 
                  [p1, p2, p3, p4, p5, p6, p7], 
                  [r1, r2, r3, r4, r5, r6, r7], 
                  [f1, f2, f3, f4, f5, f6, f7]])

    
    
    # -- Time Series Cross Validation --
    print('Normal Train Test Split: ')

    ta2, tp2, tr2, tf2, tcm2 = new_svm(data, ['tokens', 'binarised'],  time_series_split=True, time_series_split_n=2)
    ta3, tp3, tr3, tf3, tcm3 = new_svm(data, ['tokens', 'binarised'],  time_series_split=True, time_series_split_n=3)
    ta4, tp4, tr4, tf4, tcm4 = new_svm(data, ['tokens', 'binarised',], time_series_split=True, time_series_split_n=4)
    ta5, tp5, tr5, tf5, tcm5 = new_svm(data, ['tokens', 'binarised'],  time_series_split=True, time_series_split_n=5)
    ta6, tp6, tr6, tf6, tcm6 = new_svm(data, ['tokens', 'binarised'],  time_series_split=True, time_series_split_n=6)
    ta7, tp7, tr7, tf7, tcm7 = new_svm(data, ['tokens', 'binarised'], time_series_split=True, time_series_split_n=7)

    plot_barplot('time_series_n', 
                 ['2', '3', '4', '5', '6', '7'], 
                 [[ta2, ta3, ta4, ta5, ta6, ta7], 
                  [tp2, tp3, tp4, tp5, tp6, tp7], 
                  [tr2, tr3, tr4, tr5, tr6, tr7], 
                  [tf2, tf3, tf4, tf5, tf6, tf7]])


    # --- Few Shot Learning ---
    print('Few Shot 10%: ')
    fs_10_a, fs_10_p, fs_10_r, fs_10_f, fs_10_cm = new_svm(data, ['tokens', 'binarised'], few_shot=0.1)
    plot_heatmap('few_shot_10', data['thread_id'].unique().tolist(), fs_10_cm)
    
    print('Few Shot 20%: ')
    fs_20_a, fs_20_p, fs_20_r, fs_20_f, fs_20_cm = new_svm(data, ['tokens', 'binarised'],few_shot=0.2)
    plot_heatmap('few_shot_20', data['thread_id'].unique().tolist(), fs_20_cm)

    print('Few Shot 50%: ')
    fs_50_a, fs_50_p, fs_50_r, fs_50_f, fs_50_cm = new_svm(data, ['tokens', 'binarised'],few_shot=0.5)
    plot_heatmap('few_shot_50', data['thread_id'].unique().tolist(), fs_50_cm)

    # Plot bar graph for different tests
    plot_barplot('few_shot_learning', ['10%', '20%', '50%'], [[fs_10_a, fs_20_a, fs_50_a], [fs_10_p, fs_20_p, fs_50_p], [fs_10_r, fs_20_r, fs_50_r], [fs_10_f, fs_20_f, fs_50_f]])
    
    # --- Adding Noise and Perturbing Data ---

    print('Perturb X Synonym: ')
    x_syn_a, x_syn_p, x_syn_r, x_syn_f, x_syn_cm = new_svm(data, ['tokens', 'binarised'], augmentX='synonym', aug_prob=0.3)
    plot_heatmap('perturb_X_syn', data['thread_id'].unique().tolist(), x_syn_cm)

    print('Perturb X Word Deletion: ')
    x_del_a, x_del_p, x_del_r, x_del_f, x_del_cm = new_svm(data, ['tokens', 'binarised'], augmentX='word_deletion', aug_prob=0.3)
    plot_heatmap('perturb_X_del', data['thread_id'].unique().tolist(), x_del_cm)

    print('Perturb X Swap: ')
    x_swap_a, x_swap_p, x_swap_r, x_swap_f, x_swap_cm = new_svm(data, ['tokens', 'binarised'], augmentX='swap', aug_prob=0.3)
    plot_heatmap('perturb_X_swap', data['thread_id'].unique().tolist(), x_swap_cm) 

    print('Perturb X Spelling: ')
    x_spell_a, x_spell_p, x_spell_r, x_spell_f, x_spell_cm = new_svm(data, ['tokens', 'binarised'], augmentX='spelling', aug_prob=0.3)
    plot_heatmap('perturb_X_spell', data['thread_id'].unique().tolist(), x_spell_cm)

    print('Perturb X Char Replacement: ')
    x_char_a, x_char_p, x_char_r, x_char_f, x_char_cm = new_svm(data, ['tokens', 'binarised'], augmentX='char_replacement')
    plot_heatmap('perturb_X_char', data['thread_id'].unique().tolist(), x_char_cm)

    print('Perturb X Split: ')
    x_split_a, x_split_p, x_split_r, x_split_f, x_split_cm = new_svm(data, ['tokens', 'binarised'], augmentX='split')
    plot_heatmap('perturb_X_split', data['thread_id'].unique().tolist(), x_split_cm)

    print('Perturb X Antonym: ')
    x_ant_a, x_ant_p, x_ant_r, x_ant_f, x_ant_cm = new_svm(data, ['tokens', 'binarised'], augmentX='antonym')
    plot_heatmap('perturb_X_ant', data['thread_id'].unique().tolist(), x_ant_cm)

    print('Perturb X Keyboard: ')
    x_key_a, x_key_p, x_key_r, x_key_f, x_key_cm = new_svm(data, ['tokens', 'binarised'], augmentX='keyboard')
    plot_heatmap('perturb_X_key', data['thread_id'].unique().tolist(), x_key_cm)

    print('Perturb Y: ')
    y_label_a, y_label_p, y_label_r, y_label_f, y_label_cm = new_svm(data, ['tokens', 'binarised'], augmentY=True)
    plot_heatmap('perturb_X_char', data['thread_id'].unique().tolist(), y_label_cm)

    # Plot bar graph for tests on perturbed training data
    plot_barplot('training_perturb', 
                 ['X_Syn', 'X_Del', 'X_Swap', 'X_Spell', 'X_Char', 'X_Split', 'X_Ant', 'X_Key', 'Y'], 
                 [[x_syn_a, x_del_a, x_swap_a, x_spell_a, x_char_a, x_split_a, x_ant_a, x_key_a, y_label_a], 
                  [x_syn_p, x_del_p, x_swap_p, x_spell_p, x_char_p, x_split_p, x_ant_p, x_key_p, y_label_p], 
                  [x_syn_r, x_del_r, x_swap_r, x_spell_r, x_char_r, x_split_r, x_ant_r, x_key_r, y_label_r], 
                  [x_syn_f, x_del_f, x_swap_f, x_spell_f, x_char_f, x_split_f, x_ant_f, x_key_f, y_label_f]])

    # --- Hyperparameter Search ---
    for c in [1, 10, 100]:
        for g in [0.01, 0.1, 1, 10, 100]:
            print('C: ', c)
            print('Gamma: ', g)
            new_svm(data, ['tokens', 'binarised'], vectoriser= TfidfVectorizer(), kernel='linear', stratify=False, oversample='duplicate', C=c, g=g)

if __name__ == "__main__":
    main()