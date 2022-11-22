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

# FUNCTION TO FILTER DATA
# def get_proofs_by_length(proofs, length, antecedents=['all'], consequents=['all']):
#     filtered_proofs = []

#     for k,v in proofs.items():        
#         if isinstance(v[0], list) and len(v[0]) == length:
#             c = k.find(connective) # position of connective
#             ant = k[:c-1] # antecedent
#             con = k[c+3:] # consequent
#             if (ant in antecedents or 'all' in antecedents) and (con in consequents or 'all' in consequents):
#                 filtered_proofs.append(k)

#     return filtered_proofs

# def get_sub_proofs(sub_proofs, proofs):
#     '''
#     returns all sub_proofs from dictionary 'sub_proofs'
#     filtered by specific proofs from list 'proofs'
#     '''
#     filtered_sub_proofs = []

#     for p in proofs:
#         filtered_sub_proofs += sub_proofs[p]

#     return filtered_sub_proofs

# FUNCTION TO DECODE FROM A ONE-HOT ENCODING VECTOR TO PREMISES OF THE FORM 'X1 -> X2'
# def premise_decoder(V, pr):
#     d = len(V)        # length of antecedent/consequent   
#     n = len(pr)/(d*2) # number of premises
    
#     dec = []
#     for i in range(0, len(pr), d*2):
#         dec.append( prem(V[argmax(pr[i:i+d])], V[argmax(pr[i+d:i+2*d])]) )

#     return dec

# FUNCTIONS TO ENCODE DATA
# function that returns real-valued vectors of fixed size
# def embedding(X, vocabulary_size, embedding_size):
#     '''
#     neural network to encode embeddings
#     from a set of premises 'X'
#     '''
#     model = Sequential()
#     model.add(Embedding(vocabulary_size, embedding_size, input_length=X.shape[1]))
#     model.compile('rmsprop', 'mse')
#     output_array = model.predict(X)

#     return output_array.reshape(X.shape[0], X.shape[1]*embedding_size) # reshape from 3D to 2D

# # function that returns a binary of a maximal size
# def as_binary(integer, size):
#     d = len('{0:b}'.format(size))
#     bin_str = '{' + '0:0{}b'.format(d) + '}'
#     return [int(ch) for ch in bin_str.format(integer)]    

# def premise_encoder(V, list_pre, e_type='one_hot', embedding_size=10):
#     '''
#     input: a list of literals 'V', and a list of premises (list_pre)
#     output: a vector X of encoded premises (antecedent and consequent)
#             as 'e_type': {one_hot, ordinal, binary, embedding}            
#     '''
#     X = []    
#     n = len(V)
#     for premises in list_pre:
#         X1 = []
#         for p in premises: 
#             c = p.find(connective) # position of connective
#             if e_type == 'one_hot':
#                 a_evec = [0] * n # encode vector
#                 c_evec = [0] * n # encode vector                
#                 a_evec[V.index(p[:c-1])] = 1 # antecedent
#                 c_evec[V.index(p[c+3:])] = 1 # consequent
#                 X1 += a_evec + c_evec
#             elif e_type == 'ordinal' or e_type == 'embedding': # as integer (first)
#                 #X1 += [V.index(p[:c-1])+1, V.index(p[c+3:])+1] # does not include '0'
#                 X1 += [V.index(p[:c-1]), V.index(p[c+3:])] # includes '0'
#             elif e_type == 'binary':
#                 X1 += as_binary(V.index(p[:c-1])+1, n) + as_binary(V.index(p[c+3:])+1, n) # as binary
#         X.append(X1)  
#     X = array(X)
#     if e_type == 'embedding':
#         X = embedding(X, n, embedding_size)
#     return X  
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

# def vectorize(P, E, n_prem, filter_p=None):
#     '''
#     builds vectors X (inputs) and Y (labels)
#     with premises (in 'n_prem') and a conclusion
#     '''
#     X = []
#     Y = []
#     for c in P:
#         if (filter_p == None) or (c not in filter_p):
#             for p in P[c]:
#                 label = labels(p, [0]*len(E), E)
#                 if sum(label) in n_prem or 'all' in n_prem:
#                     X.append(E + [c])
#                     Y.append(label)
#     return X, Y
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


# def as_vectors(KB, samples=['all'], encode=True, n_prem=['all'], filter_p=None, e_type='one_hot', embedding_size=10, sequences=False):
#     V = KB['Literals']
#     X = []
#     Y = []
#     for k in KB:
#         if k.startswith('sample') and (('all' in samples) or (int(k[7:]) in samples)):
#             s = KB[k]
#             X1, Y1 = vectorize(s['Proofs'], s['Premises'], n_prem, filter_p)  
#             X += X1
#             Y += Y1
#     if encode:
#         if sequences:        
#             X = as_sequences(V, X) # encoding for RNN
#         else:
#             X = premise_encoder(V, X, e_type, embedding_size) # encoding for MLP
#     Y = array(Y)
#     return X, Y
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

# def encode_file(filename, samples=['all'], n_prem=['all'], filter_p=None, e_type='one_hot', embedding_size=10, sequences=False):
#     KB = load_data(filename)
#     X, Y = as_vectors(KB, samples, True, n_prem, filter_p, e_type, embedding_size, sequences)
#     return X, Y