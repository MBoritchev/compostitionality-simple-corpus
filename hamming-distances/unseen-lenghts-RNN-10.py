###############################################################################################################################
# This script plots the average Hamming distances for the unseen length experiment, RNN, maximal length of unseen proof = 10. #
###############################################################################################################################

# To run the script, change the path_exp_file to the path to the cloned github repository on your computer.

path_exp_file = "/Users/mboritchev/Desktop/Research/Warsaw/Hybrid-Models-of-Natural-Reasoning/"

###Parser###

def line_to_vector(s_line): # This function takes a string of values and outputs an array that contains the float values of the string.
    vector = []
    stripped_line = s_line.strip() # First, we remove the extra spaces left and right side from the values in the string.
    if stripped_line.startswith('['):
        stripped_line = stripped_line[1:]
    elif stripped_line.endswith(']'):
        stripped_line = stripped_line[:len(stripped_line)-1]
    splitted_line = stripped_line.split()
    for i in range(len(splitted_line)):
        vector.append(float(splitted_line[i]))
    return(vector)

def string_to_list(str): # This function takes a string version of an array and converts it to an array.
    line_string = []
    line_float = []
    str = str[1:]
    str = str[:(len(str)-1)]
    line_string = list(str.split(","))
    for i in range(len(line_string)):
        line_float.append(float(line_string[i].strip()))
    return line_float

def path_to_vectors(s_file): # This function takes the path to our log file and outputs the array of vectors corresponding to 1) the prediction values, 2) the predicted vector, 3) the actual vector.
    resultat = []
    flag = 0
    pred_values = []
    p_line_stripped = ""
    p_current = []
    a_current = []
    p_resultat = []
    a_resultat = []
    f = open(s_file,"r")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('P'):
            p_line_stripped = line.strip()[12:]
            if p_line_stripped.startswith('['):
                pred_values.append(string_to_list(p_line_stripped))
            else:
                pred_values.append(-1)
            if flag == 2:
                a_resultat.append(a_current)
                a_current = []
            flag = 1
        elif line.startswith('A'):
            if flag == 1:
                p_resultat.append(p_current)
                p_current = []
            flag = 2
        else:
            if flag == 1:
                p_current.extend(line_to_vector(line))
            elif flag == 2:
                a_current.extend(line_to_vector(line))
    a_resultat.append(a_current)
    resultat.append(pred_values)
    resultat.append(p_resultat)
    resultat.append(a_resultat)
    f.close()
    return(resultat)

def call_parser_unseen(model_name,n1,n2,m): # This function calls the parser on the logs corresponding to the output of the model model_name and unseen proofs of length n1 to n2.
    results = []
    path_n = ""
    path_MLP = path_exp_file + "experiment_1/productivity/logs_MLP/"
    path_RNN = path_exp_file + "experiment_1/productivity/logs_RNN/" 
    if m > n2 or m < n1:
        print("This log is not available.")
    if model_name == "MLP":
        path_n = path_MLP + "unseen_" + str(n1) + "-" + str(n2) + "/log_results_" + str(m) + ".txt" #logs_MLP/unseen_0-2/log_results_1.txt
    elif model_name == "RNN":
        path_n = path_RNN + "unseen_" + str(n1) + "-" + str(n2) + "/log_results_" + str(m) + ".txt"
    else:
        print("This is not a valid model name.")
    results = path_to_vectors(path_n)
    return results

###Available logs###

def log_triples10():
    n1 = 0
    n2 = 10
    parameters = []
    for i in range(6,11):
        n1 = i
        for m in range(n1, n2 + 1):
            parameters.append([n1,n2,m])
    return parameters


###Computing average Hamming distance for a given log###

from scipy.spatial import distance
from scipy.spatial.distance import hamming
import numpy as np 


def round_vector(vect): # This function rounds up all values in the argument vector.
    rounded_p_v = []
    for i in range(len(vect)):
        rounded_p_v.append(np.round(vect[i])) #rounded_p_v.append(np.round(p_vector[i]))
    return(rounded_p_v)


def ham_dist(v1,v2): # This function computes the hamming distance between 2 vectors fo same length 
    if len(v1) != len(v2):
        print("The two vectors are of different lenght.")
    else:
        return hamming(v1, v2)*len(v1)


def average_ham_dist(vv1,vv2): # This function computes the average hamming distance between 2 vectors of vectors of same length
    av_h = 0
    if len(vv1) != len(vv2):
        print("The two vectors are of different length.")
    else:
        l = len(vv1)
        for i in range(l):
            av_h = av_h + ham_dist(vv1[i], vv2[i])
        return (av_h/l)


def average_ham_log(model_name,n1,n2,m): # This function outputs the average hamming distance for a given log
    vis_v = call_parser_unseen(model_name,n1,n2,m)
    p_vect = vis_v[1]
    a_vect = vis_v[2]
    ham = 0
    round_p_vect = round_vector(p_vect)
    round_distance = average_ham_dist(round_p_vect,a_vect)
    return round_distance

###Visualization###

import matplotlib.pyplot as plt 

def all_average_ham(model):
    x_array = log_triples10()
    z_val = []
    for i in range(len(x_array)):
        z_val.append(average_ham_log(model,x_array[i][0],x_array[i][1],x_array[i][2]))
    return z_val

def visualisation_fct_average_ham(model):
    x_array = log_triples10() # contains all possible n1,n2,m triples, defining all existing logs for given model
    z_val = all_average_ham(model)
    plt.bar(range(len(x_array)), z_val, 0.4, label = 'Hamming distance')
    plt.xticks(range(len(x_array)), x_array,rotation = 90)
    plt.ylabel("Distance prediction/actual")
    plt.legend()
    plt.show()


visualisation_fct_average_ham("RNN")

