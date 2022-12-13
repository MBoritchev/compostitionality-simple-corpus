# SIMPLE CORPUS
# =============
# MULTI-LABEL APPROACH: DATA ENCODING
# ===================================
from numpy import argmax, array
from json import load
from keras.models import Sequential
from keras.layers import Embedding
from data_generation import prem, connective

from random import shuffle

def premise_encoder(V, list_pre):
    '''
    input: a list of literals 'V', and a list of premises (list_pre)
    output: a vector X of encoded premises (antecedent and consequent)
            as 'e_type': {one_hot, ordinal, binary, embedding}            
    '''
    X = []    
    n = len(V)
    for premises in list_pre:
        X1 = []
        for p in premises: 
            c = p.find(connective) # position of connective
            a_evec = [0] * n # encode vector
            c_evec = [0] * n # encode vector                
            a_evec[V.index(p[:c-1])] = 1 # antecedent
            c_evec[V.index(p[c+3:])] = 1 # consequent
            X1 += a_evec + c_evec
        X.append(X1)  
    X = array(X)

    return X  


def as_sequences(V, list_pre):    
    seq_X = []
    for x in list_pre:
        seq_X.append(premise_encoder(V, [[p] for p in x]))

    return array(seq_X)    

def labels(proof, Y, E):
    '''
    labels the premises used in a proof
    with the number 1 in a vector 'Y'
    '''
    if proof != 'false':
        for p in proof:
            Y[E.index(p)] = 1
    return Y

def vectorize(P, E, n_prem):
    '''
    builds vectors X (inputs) and Y (labels)
    with premises (in 'n_prem') and a conclusion
    '''
    X = []
    Y = []
    for c in P:
        for p in P[c]:
            label = labels(p, [0]*len(E), E)
            if sum(label) in n_prem or 'all' in n_prem:
                X.append(E + [c])
                Y.append(label)

    return X, Y

def as_vectors(KB, samples=['all'], encode=True, n_prem=['all'], sequences=False):
    V = KB['Literals']
    X = []
    Y = []
    for k in KB:
        if k.startswith('sample') and (('all' in samples) or (int(k[7:]) in samples)):
            s = KB[k]
            X1, Y1 = vectorize(s['Proofs'], s['Premises'], n_prem)  
            X += X1
            Y += Y1
    if encode:
        if sequences:        
            X = as_sequences(V, X) # encoding for RNN
        else:
            X = premise_encoder(V, X) # encoding for MLP
    Y = array(Y)
    return X, Y

# MAIN FUNCTIONS TO LOAD AND ENCODE JSON FILES
def load_data(filename):
    f = open(filename, 'r')
    KB = load(f)
    f.close()
    
    return KB