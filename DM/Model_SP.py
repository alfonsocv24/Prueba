#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:28:49 2023

@author: alfonsocabezonvizoso
"""

'''Sequential Properties for CycMPDB data base'''

import time
start = time.time()
from encoder_PCA import Encoder
encoder = Encoder(properties='All')
from UP import CyclicPeptide
CP = CyclicPeptide()
import tensorflow as tf
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import argparse
import pickle

def geometric_mean_score(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr * tnr)**0.5

'''Define user inputs'''
###############################################################################
parser = argparse.ArgumentParser(description='''ML code''')
parser.add_argument('-ds', '--DATASET', help='Dataset name, AllPep or L67',
                    action='store', type=str, default = 'L67',
                    choices=['L67'])

args = parser.parse_args()
###############################################################################

dataset = args.DATASET # get dataset name
file = f'CycPeptMPDB_{dataset}.csv'
#Load data

data = pd.read_csv(file, header = 0)
#Assign a 1 to permeable if permeability < -6, otherwise assign 0
data['Permeable'] = np.where(data['Permeability'] < -6, 1, 0 )

# Eliminate repeated sequences
data.drop_duplicates(subset='Sequence',keep = False,  inplace = True)

X = data['Sequence'].to_numpy()# Get sequences
y = data['Permeable'].to_numpy() #Our target
train_portion = 0.8
val_portion = 0.2

# Model builder function
def model_builder(hp):
    '''
    This function creates a Machine Learning model based on sequential
    properties using keras
    Layers:
        1. Two stacked Convolutional Layers
        2. LSTM
        3. Multi Layer Perceptron
        4. Output layer with sigmoid activation.

    Parameters
    ----------
    hp : keras tuner object
        Object passed internally by the tuner.

    Returns
    -------
    model : keras.src.models.sequential.Sequential
        Sequential Properties model.

    '''
    model = Sequential()
    #Add convolutional layer as input
    hp_filters = hp.Int('filters', min_value = 32, max_value = 256, step = 32)
    #The tuner will test 32 and 64 for the CNN
    model.add(Conv1D(hp_filters, 3, input_shape = (15,208), kernel_initializer = "he_normal"))
    #Add a hidden CNN layer
    hp_filters2 = hp.Int('filters2', min_value = 16, max_value = 144, step = 16)
    #The tuner will test 32 and 64 for the CNN
    model.add(Conv1D(hp_filters2, 3, kernel_initializer = "he_normal"))
    #The second number denotes "the length of the patterns we want to learn from
    #Add LSTM layer
    hp_units = hp.Int('units_LSTM', min_value = 64, max_value = 256, step = 64)
    model.add(LSTM(hp_units, activation = 'relu', return_sequences = False))
    #Add dropout layer. This fights overfitting  by setting input units 0 with a rate
    # model.add(Dropout(0.2))
    #Add first layer of the fully connected part
    hp_neurons = hp.Int('units_Dense', min_value=16, max_value = 208, step = 16)
    model.add(Dense(hp_neurons, activation = 'relu'))
    #Add a second dropout layer
    # model.add(Dropout(0.2))
    #Add a hidden layer
    hp_neurons2 = hp.Int('units_Dense2', min_value = 4, max_value = 132, step = 16)
    model.add(Dense(hp_neurons2, activation = 'relu'))
    #Add output layer
    model.add(Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3, 1e-4])
    #Here the tuner will test the different values we propose
    #define optimizer applying tuner for learning rate
    opt = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
    #Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model

num_folds = 10
#Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
# Load mutations
with open('redmut_dict.pkl', 'rb') as f:
    dict_have_aa = pickle.load(f)

# Define stratified K-fold cross validator
kfold = RepeatedStratifiedKFold(n_repeats=3, n_splits = num_folds, random_state=42)


fold_no = 1 
final_lenght = []
for train, test in kfold.split(X, y):
    #Create validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X[train],
                                                    y[train],
                                                    test_size =val_portion,
                                                    random_state=42,
                                                    stratify = y[train])
    
    #Train
    TRAIN = X_train
    TRAIN_length = len(TRAIN)
    TRAIN_target = y_train
    seq_muts = []
    seq_muts_target = []
    for idx in range(len(TRAIN)):
        seq = TRAIN[idx]
        target = TRAIN_target[idx]
        lst_permutations = CP.cyclic_permutations(seq)
        for sequence in lst_permutations:
            if sequence not in seq_muts:
                seq_muts.append(''.join(sequence))
                seq_muts_target.append(target)
    TRAIN = np.array(seq_muts.copy())
    TRAIN_target = np.array(seq_muts_target.copy())
    dict_seqs = {}
    only_mut = {}
    for i in range(0,len(TRAIN)):
        seq = TRAIN[i] 
        dict_seqs[seq] = []
        if seq not in dict_have_aa.keys():
            continue
        mutations = np.array(dict_have_aa[seq][0])
        if len(mutations) == 0:
            continue
        tar_mut = dict_have_aa[seq][1][0]
        for mut_seq in mutations:
            dict_seqs[mut_seq] = []
            only_mut[mut_seq] = []
    for i in range(0,len(TRAIN)):
        seq = TRAIN[i]
        dict_seqs[seq].append(TRAIN_target[i])
        if seq not in dict_have_aa.keys():
            continue
        mutations = np.array(dict_have_aa[seq][0])
        if len(mutations) == 0:
            continue
        target_class = dict_have_aa[seq][1][0]
        for mut_seq in mutations:
            dict_seqs[mut_seq].append(target_class)
            only_mut[mut_seq].append(target_class)

    sequences = list(dict_seqs.keys())
    sequences = np.array(sequences)
    final_TRAIN = []
    final_TRAIN_target = []
    for seq in sequences:
        if len(set(dict_seqs[seq])) == 1:
            #Only keep sequences that are not repeated with different target class
            final_TRAIN.append(seq)
            final_TRAIN_target.append(dict_seqs[seq][0])
    # Add original sequences so we don't miss any of them
    final_dict = dict(zip(final_TRAIN, final_TRAIN_target))
    for ind, seq in enumerate(TRAIN):
        final_dict[seq] = TRAIN_target[ind]
    final_TRAIN = np.array(list(final_dict.keys()))
    final_TRAIN_target = np.array(list(final_dict.values()))
    TRAIN_target = np.array(final_TRAIN_target)
    TRAIN, _ = encoder.encode(sequences = final_TRAIN, length = 15, stop_signal = False)
    #Test
    TEST = X[test]
    TEST_length = len(TEST)
    TEST_target = y[test]
    TEST, _ = encoder.encode(sequences = TEST, length = 15, stop_signal = False)
    #Validation
    VAL = X_val
    VAL_length = len(VAL)
    VAL_target = y_val
    VAL, _ = encoder.encode(sequences = VAL, length = 15, stop_signal = False)
    
    # Instantiate the tuner for hyperband search
    tuner = kt.Hyperband(model_builder, objective = 'val_accuracy',
                          max_epochs = 80, factor = 3, project_name=f'Tuner_fold{fold_no}', 
                          overwrite=False, seed = 42)
    # Create early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5)

    early_stopping_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=35, restore_best_weights=True)
    
    # Start hyperparameter search
    tuner.search(TRAIN, TRAIN_target, epochs = 2000, validation_data = (VAL, VAL_target),
                  callbacks = [early_stopping], batch_size = 32)
    #Get optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] #returns top performing hyp
    model = tuner.hypermodel.build(best_hps) #automatically builds the model with the best parameters
    ##TRAIN the model
    history = model.fit(TRAIN, TRAIN_target, epochs = 2000,
              validation_data=(VAL, VAL_target), batch_size= 32, verbose = 1,
              callbacks=[early_stopping_train])
    # Save trained model
    model.save(f'OptimizedModel_fold{fold_no}.keras')

    #EVALUATION
    scores_train = model.evaluate(TRAIN, TRAIN_target, verbose = 0)
    scores = model.evaluate(TEST, TEST_target, verbose = 0)
    scores_val = model.evaluate(VAL, VAL_target, verbose = 0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    data_2_plot = [208, fold_no, scores_val[1]*100, scores[1]*100,scores_train[1]*100]
    sep = '    '
    # COMPUTE METRICS
    y_pred = model.predict(TEST)
    new_y = []
    #Round values to be 0 or 1
    for i in y_pred:
        for j in i:
            new_y.append(round(j))
    y_pred = np.array(new_y)
    confusion_TEST = confusion_matrix(TEST_target, y_pred)
    precision_TEST = precision_score(TEST_target, y_pred)
    f1_TEST = f1_score(TEST_target, y_pred)
    recall_TEST = recall_score(TEST_target, y_pred)
    roc_auc_TEST = roc_auc_score(TEST_target, y_pred)
    matthews_TEST = matthews_corrcoef(TEST_target, y_pred)
    geom_mean_TEST = geometric_mean_score(TEST_target, y_pred)
    with open(f'{dataset}_split_lengths.dat', 'a') as f:
        if fold_no == 1:
            f.write('fold_no'+sep+'TRAIN'+sep+'TEST'+sep+'VAL\n')
        f.write(f'{data_2_plot[1]}{sep}{TRAIN_length}{sep}{TEST_length}{sep}{VAL_length}\n')
        f.close()

    with open(f'Metrics_{dataset}.dat', 'a') as f:
        if fold_no == 1:
            f.write('Fold'+sep+'True00'+sep+'False01'+sep+'False10'+sep+'True11'+sep+'Precision'+sep+'f1'+sep+'Recall'+sep+'Roc_auc'+sep+'Matthews'+sep+'Geom_mean\n')
        f.write('{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'.format(fold_no, sep, confusion_TEST[0][0],sep,
                                                                  confusion_TEST[0][1],sep,
                                                                  confusion_TEST[1][0],sep,
                                                                  confusion_TEST[1][1],sep,
                                                                  precision_TEST,sep,f1_TEST,
                                                                  sep,recall_TEST,sep,roc_auc_TEST,
                                                                  sep,matthews_TEST,sep,geom_mean_TEST))
        f.close()

    with open(f'Accuracy_{dataset}.dat', 'a') as f:
        if fold_no == 1:
            f.write('n_Features'+sep+'fold_no'+sep+'Acc_validation'+sep+'Acc_test'+sep+'Acc_train\n')
        f.write(f'{data_2_plot[0]}{sep}{data_2_plot[1]}{sep}{data_2_plot[2]}{sep}{data_2_plot[3]}{sep}{data_2_plot[4]}\n')
        f.close()
    #Increase fold number
    fold_no += 1
end = time.time()
print(f'Time needed: {end - start}s')
    
