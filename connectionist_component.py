# SIMPLE CORPUS
# =============
# MULTI-LABEL APPROACH: NEURAL NETWORK
# ====================================
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras import initializers
from sklearn.model_selection import train_test_split

from data_encoding import premise_decoder

# STATISTICS
# create a log of results (proofs of length 1)
def log_results(all_pred, Y, n):
    f = open('log_results_{}.txt'.format(n), 'w') # LOG

    for i in range(len(Y)):
        actual_label = Y[i]
        pr_used = sum(actual_label)
        if pr_used == n:             
            #p = np.round(all_pred[i][np.argmax(actual_label)], decimals=2)
            if pr_used > 0:
                all_max = np.argwhere(actual_label == np.amax(actual_label)).flatten().tolist()
                p = [np.round(all_pred[i][j], decimals=2) for j in all_max]
                f.write('Prediction: {}\n'.format(p)) # prediction value(s)
            else:
                f.write('Prediction: false\n') 
            f.write(str(np.round(all_pred[i], decimals=2))) # prediction vector
            f.write('\n')
            f.write('Actual label:\n')
            f.write(str(actual_label)) # label vector (actual answer)
            f.write('\n') 

    f.close()

# Show statistics and results
def stats(all_pred, Y_train, Y_test):
    statistics = {}
    for y in Y_train:
        pr_used = sum(y)
        if pr_used in statistics.keys():
            statistics[pr_used][0] += 1
        else:
            statistics[pr_used] = [1, 0, 0]

    for i in range(len(Y_test)):
        prediction = np.round(all_pred[i])
        actual_label = Y_test[i]
        result = int(all(prediction == actual_label))
        pr_used = sum(actual_label)    

        if pr_used in statistics.keys():
            statistics[pr_used][1] += 1
            statistics[pr_used][2] += result
        else:
            statistics[pr_used] = [0, 1, result]

    #log_results(all_pred, Y_test) # create a log file text

    #print statistics
    print('Train/validation data =', len(Y_train), '\nTests =', len(Y_test))
    overall = np.round(100 * sum([statistics[i][2] for i in statistics.keys()]) / len(Y_test), decimals=2)
    overall_p = np.round(100 * sum([statistics[i][2] for i in statistics.keys() if i != 0]) / len([y for y in Y_test if sum(y) > 0]), decimals=2)
    print('Overall correct tests = {}%'.format(overall))
    print('Overall correct valid tests = {}%'.format(overall_p))
    print()
    print('Used', 'Train', 'Test', 'Correct', 'Percentage', sep='\t')
    print('------------------------------------------')
    for i in sorted(statistics):
        train = statistics[i][0]
        test = statistics[i][1]
        correct = statistics[i][2]
        if test == 0: 
            percentage = 0
        else: 
            percentage = np.round((correct*100)/test, decimals=2)        
        print(i, train, test, correct, '{}%'.format(percentage), sep='\t')

# PLOT LOSS AND ACCURACY HISTORY
def plot_loss_accuracy(loss, val_loss, acc, val_acc):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('Log Loss and Accuracy over iterations')
    
    # add_subplot(rows, columns, index)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss)
    ax.plot(val_loss)
    ax.grid(True)
    ax.set(xlabel='epochs', title='Log Loss')
    ax.legend(['train', 'validation'], loc='upper right')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(acc)
    ax.plot(val_acc)
    ax.grid(True)
    ax.set(xlabel='epochs', title='Accuracy')
    ax.legend(['train', 'validation'], loc='lower right')
    
    plt.show()

# DEFINITION AND TRAINING
# Model types: {Multilayer Perceptron: 'MLP', (Simple) Recurrent Neural Network: 'RNN'}
def fit_model(X, Y, model_type = 'MLP', **options): 
    # CHECK OPTIONS
    #plot=True, model_name=None, n_perceptrons=500, h_layers=3, n_epochs=30, batch_size=32, split = [0.5, 0.25, 0.25]

    # plot training history
    if 'plot' in options.keys():
        plot = options['plot'] # If False then do not plot results
    else:
        plot = True # default value

    # write the model to a file
    if 'model_name' in options.keys():
        model_name = options['model_name'] #  name (str) to save the model
    else:
        model_name = None # default value

    # number of perceptrons per layer
    if 'n_perceptrons' in options.keys():
        n_perceptrons = options['n_perceptrons'] # integer > 0
    else:
        n_perceptrons = 500 # default value

    # number of hidden layers
    if 'h_layers' in options.keys():
        h_layers = options['h_layers'] # integer > 0
    else:
        h_layers = 3 # default value

    # number of epochs
    if 'n_epochs' in options.keys():
        n_epochs = options['n_epochs'] # integer > 0
    else:
        n_epochs = 30 # default value

    # batch size
    if 'batch_size' in options.keys():
        batch_size = options['batch_size'] # integer > 1
    else:
        batch_size = 32 # default value: update parameters every (all_samples/32) samples

    # percentages of the split
    if 'split' in options.keys():
        split = options['split'] # list of percentages for training/validation/test
        assert sum(split) == 1, 'The sum of all values in "split" must be exactly 1'
    else:
        split = [0.5, 0.25, 0.25] # default value

    # stratification
    if 'stratify' in options.keys() and options['stratify'] == False: # do not stratify split
        strat = None
    else:
        strat = [sum(y) for y in Y] # default option    

    # MLP activation function for hidden layers:
    if '    ' in options.keys():
        activation_function = options['activation_function']
    else:
        activation_function = 'relu'

    # optimizers: SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
    if 'optimizer' in options.keys():
        opt = options['optimizer']
    else:
        opt = 'Adam'

    # DATA SPLIT    
    val_size = split[1] / (1 - split[2])
    if split[2] == 0:
        X_train = X
        Y_train = Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split[2], stratify = strat)

    # MODEL DEFINITION    
    model = Sequential() # instantiation of NN

    # hidden layers according to model type
    # Multilayer Perceptron
    if model_type == 'MLP':
        model.add(Dense(n_perceptrons, input_dim = X.shape[1], activation=activation_function)) # hidden layer 1
        for i in range(h_layers-1):
            model.add(Dense(n_perceptrons, activation=activation_function)) # hidden layer(s)

    # Simple Recurrent Neural Network (2 layers)
    elif model_type == 'RNN':
        model.add(SimpleRNN(n_perceptrons, input_shape = X.shape[1:], return_sequences=True))
        model.add(SimpleRNN(n_perceptrons))

    # output layer
    model.add(Dense(Y.shape[1], activation='sigmoid')) 

    # MODEL COMPILATION AND TRAIN
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model on the dataset
    H = model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split = val_size)

    if model_name != None:
        model.save(model_name)
    if plot:
        plot_loss_accuracy(H.history['loss'], H.history['val_loss'], H.history['accuracy'], H.history['val_accuracy']) 

    if split[2] > 0:
        stats(model.predict(X_test), Y_train, Y_test)

    return model

# function to print statistics from tested data
def show_results(all_pred, Y):
    results = {}

    for i in range(len(Y)):
        prediction = np.round(all_pred[i])
        actual_label = Y[i]
        result = int(all(prediction == actual_label))
        pr_used = sum(actual_label)    

        if pr_used in results.keys():
            results[pr_used][0] += 1
            results[pr_used][1] += result
        else:
            results[pr_used] = [1, result]

    #log_results(all_pred, Y) # create a log file text

    # print statistics from dictionary    
    print('Total number of tests performed =', len(Y))

    overall = np.round(100 * sum([results[i][1] for i in results.keys()]) / len(Y), decimals=2)
    overall_p = np.round(100 * sum([results[i][1] for i in results.keys() if i != 0]) / len([y for y in Y if sum(y) > 0]), decimals=2)
    print('Overall correct tests = {}%'.format(overall))
    print('Overall correct valid tests = {}%'.format(overall_p))

    print()
    print('Pr_used', 'Tested', 'Correct', 'Percentage', sep='\t')
    print('----------------------------------')
    for i in sorted(results):
        test = results[i][0]
        correct = results[i][1]
        if test == 0: 
            percentage = 0
        else: 
            percentage = np.round((correct*100)/test, decimals=2)        
        print(i, test, correct, '{}%'.format(percentage), sep='\t')

    return results