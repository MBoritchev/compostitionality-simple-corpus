# SIMPLE CORPUS
# =============
# MULTI-LABEL APPROACH: DATASET GENERATOR
# =======================================
from random import randint, random, shuffle
from json import dump
from os import getcwd

# FUNCTIONS TO CREATE THE PROOFS
def prem(i, t):
    '''
    takes an initial 'i' and a target 't' literals
    and returns a string in form of a premise: 'i -> t'
    '''
    return '{} {} {}'.format(i, connective, t)

def path_to_premises(paths):
    '''
    take a list of paths and returns
    a list of premises (pairs)
    '''
    proofs = []
    for path in paths:
        proof = []
        for i in range(len(path)-1):
            proof.append(prem(path[i], path[i+1]))
        proofs.append(proof)        
    return proofs

def find_all_paths(E, start, end, path=[]):
    '''
    Takes a set of edges E, and returns a list 
    of paths from nodes 'start' to 'end'
    '''
    path = path + [start]
    if start == end:
        return [path]
    if start not in set([n1 for (n1,n2) in E]):
        return []
    paths = []
    for node in [n2 for (n1,n2) in E if n1 == start]:
        if node not in path:
            newpaths = find_all_paths(E, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def find_all_sub_paths(paths):
    path = paths[0] # because one solution (so far)
    n = len(path)
    sub_paths = []
    #sub_paths_length = 3 # 3 = proofs of size 2 or greater
    sub_paths_length = 2 # 2 = all proofs

    for i in range(n):
        for j in range(i+sub_paths_length, n+1):
            #if len(path[i:j]) < n: # do not include the path itself
            sub_paths.append(path[i:j])        
    #for s in sub_paths:
    #    print(s)
    return [(prem(s[0], s[-1]), len(s)-1) for s in sub_paths]    

def pairs(V):
    '''
    returns all possible pairs from 
    a set of literals 'V'
    '''
    all_pairs = []
    for v in V:
        all_pairs += [(v,n) for n in V if v!=n]
    return all_pairs

def create_proofs(G, V, n_prem=['all']):
    '''
    returns a dictionary with conclusions as keys and
    a list (or lists) of premises that entail the conclusion
    '''
    proofs = {}
    sproofs = {}
    conclusions = pairs(V)
    for i,t in conclusions:
        paths = find_all_paths(G, i, t)                
        p = path_to_premises(find_all_paths(G, i, t))
        if p != []:
            if 'all' in n_prem or len(p[0]) in n_prem:                
                proofs[prem(i,t)] = path_to_premises(paths)
                sproofs[prem(i,t)] = find_all_sub_paths(paths)
        else: # it is not a proof
            if 'all' in n_prem or 0 in n_prem:
                proofs[prem(i,t)] = ['false']

    return proofs, sproofs

# CREATION OF TREE-LIKE STRUCTURES     
# def dag_from_file(filename):
#     f = open(filename, 'r')
#     g = f.readlines()
#     f.close()

#     i = 1
#     levels = []
#     literals = []
#     E = []
#     for line in g:    
#         if line.find(',') > 0:
#             p = line.find('[')
#             curr_literals = line[:p].strip().split(', ')        
#             levels += [(i, x) for x in curr_literals]
#             literals += curr_literals
#             i += 1
#         else:
#             p = line.find('->')
#             if p > 0:
#                 E.append((line[:p], line[p+2:-2]))

#     return E, levels, literals

def one_path(cn, GP, E):
    '''
    inputs: a node 'cn', a set of parent nodes  'GP'
            a set of edges 'E' (tree)
    output: True, if there is at most one path to 'cn'
            from all nodes in 'GP'
            False, otherwise
    '''
    for pn in GP:
        if len(find_all_paths(E, pn, cn)) > 1:
            return False
    return True

def random_tree(min_n, max_n, L, P):
    '''
    creates a tree of 'L' levels, each level with N nodes (literals)
    that are computed randomly, such that: 'min_n' <= N <= 'max_n'
    '''

    # creates a list of levels and a list of literals 'X_c'
    c = 1
    levels = []
    literals = []
    for i in range(1, L+1):
        for j in range(randint(min_n, max_n)):
            levels.append((i, 'X'+str(c)))            
            literals.append('X'+str(c))
            c +=1

    # creates a tree
    E = [] # set of edges
    for i in range(1, L):    
        c_level = [n for (r,n) in levels if r==i]   # list of nodes from current level
        n_level = [n for (r,n) in levels if r==i+1] # list of nodes from next level        
        # add edges (premises) with a probability 'P'
        for cn in c_level:
            for nn in n_level:            
                if P > random() and one_path(nn, [n for (r,n) in levels if r < i], E + [(cn,nn)]):
                    E.append((cn,nn))        

    return E, levels, literals

# shuffling the literals
def literals_shuffling(E, levels, literals):
    shuffle(literals)
    new_E = []
    new_levels = []
    for i,t in E:
        new_E.append((literals[int(i[1:])-1], literals[int(t[1:])-1]))
        #print(int(i[1:]),int(t[1:])) 
    for le,li in levels:
        new_levels.append((le, literals[int(li[1:])-1]))
    
    return new_E, new_levels, literals

# KB information functions
def KB_summary(proofs, Li, Pr, samples):
    '''
    returns a dictionary with a detailed summary 
    of a single knowledge base and all its proofs
    '''
    S = {'Number of samples': samples, 'Number of literals': Li, 'Number of premises': Pr, 'Total proofs': 0, 'Valid proofs': 0}
    for c in proofs.keys():
        if isinstance(proofs[c][0], list):
            pr_used = len(proofs[c][0])
            S['Valid proofs'] += 1
        else:
            pr_used = 0
        if pr_used in S.keys():
            S[pr_used] += 1
        else: 
            S[pr_used] = 1
        S['Total proofs'] += 1
    return S

def sub_proofs_summary(KB, KB_filter=None):        
    summary = {}
    for s in KB.keys():
        if s.startswith('sample'):
            sp = KB[s]['Sub_proofs']
            if KB_filter:
                sp_filter = KB_filter[s]['Proofs']

            for k,i in sp.items():                
                if (KB_filter==None or k in sp_filter) and len(i) > 1: # if it is a proof of length 2 or greater  
                    cat = max([l for _,l in i])
                    if cat in summary.keys():
                        summary[cat][0] += 1
                    else:
                        summary[cat] = [1, 0, 0]
                    for c,l in i:
                        if (KB_filter==None) or (c in sp_filter):
                            summary[cat][1] += 1
                            summary[cat][2] += l
    return summary


# def sub_proofs_summary1(sp, filter=None):    
#     print('Proof\t       ', 'S(P)   ', 'R(P)\t')
#     #for k,i in sp.items():        
#     for k,i in sorted(sp.items(), key=lambda item: len(item[1]), reverse=True):
#         if (filter == None) or (k in filter):            
#             print(k, len(i), sum([p[1] for p in i]), sep='\t')

# def sub_proofs_summary2(sp):
#     #sp = kb['sample 1']['Sub_proofs']
#     overlapping = {}
#     for k in sp.keys():
#         overlapping[k] = []
#         for m in sp.keys():
#             if k in sp[m]:
#                 overlapping[k].append(m)

#     print('Proof\t', 'Frequency', sep='\t')
#     print('-------------------------')
#     for k, v in sorted(overlapping.items(), key=lambda item: len(item[1]), reverse=True):
#         print(k, len(v), sep='\t')

# FUNCTIONS TO WRITE THE DATA TO A FILE
def save_data(KB, filename):    
    '''
    writes the dictionary 'KB' to a json file
    '''
    with open(filename,'w') as file:
        dump(KB, file)
        file.close()        

# def create_dot(E, levels, filename, w_color):
#     '''           
#     writes a graphviz script (DOT language) 
#     from a set of edges E to a file 'filename'
#     '''    
#     L = max([l for (l,_) in levels])
#     f = open(filename, 'w')    
#     f.write('digraph G {\n')
#     if w_color:
#         red = Color("red")
#         colors = list(red.range_to(Color("green"), L))
#         f.write(' {\n node [style=filled]\n')
#         for i in range(1, L+1):
#             f.write(' {} [color="{}"]\n'.format(', '.join(str(e) for e in [n for (r,n) in levels if r==i]), colors[i-1]))
#         f.write(' }\n')

#     for n1, n2 in E:
#         f.write('{}->{};\n'.format(n1,n2))
#     f.write('}')
#     f.close() 
def create_dot(E, levels, filename):
    '''           
    writes a graphviz script (DOT language) 
    from a set of edges E to a file 'filename'
    '''    
    L = max([l for (l,_) in levels])
    f = open(filename, 'w')    
    f.write('digraph G {\n')

    for n1, n2 in E:
        f.write('{}->{};\n'.format(n1,n2))
    f.write('}')
    f.close() 


# FUNCTIONS TO SPLIT DATA (TRAIN/TEST) FROM A KB
# STRATIFICATION BY LENGHT OF PROOFS
def stratification(P):
    strat_data = {} 
    for k,i in P.items():
        if i[0] == 'false':
            n = 0
        else:
            n = len(i[0])
        if n not in strat_data.keys():
            strat_data[n] = [(k, i)]
        else:
            strat_data[n] += [(k, i)]
    return strat_data

def stratified_split(P, perc={}, test_perc=0.25):
    '''
    input: dictionary of proofs 'P': {c: [path1, path2, ...], ...},
           perc: {0: p_0, 1: p_1, ..., n: p_n}
           where 'n' is the max number of used premises and each
           'p_i' is the percentage used as test data (default=0.25)
    output: splitted stratified dictionaries (for training and test)
    '''
    S = stratification(P)    
    train = []
    test = []
    for k in S.keys():
        if k in perc:
            p = perc[k] # custom percentage
        else:
            p = test_perc # default percentage

        lte = round(len(S[k]) * p) # length of test data
        ltr = len(S[k]) - lte      # length of train data
        shuffle(S[k]) # shuffle data

        # split into train and test:
        train += S[k][:ltr]
        if lte > 0: 
            test += S[k][-lte:]
        
    return {k:i for (k,i) in train}, {k:i for (k,i) in test}

def split_knowledge_base(KB, filename=None, perc={}, test_perc=0.25):
    literals = KB['Literals']

    KB_train = {'Literals': sorted(literals)}
    KB_test = {'Literals': sorted(literals)}

    for k in KB:
        if k.startswith('sample'):
            premises = KB[k]['Premises']
            train_proofs, test_proofs = stratified_split(KB[k]['Proofs'], perc, test_perc)

            KB_train[k] = {'Premises': premises, 'Proofs': train_proofs}
            KB_test[k] = {'Premises': premises, 'Proofs': test_proofs}

    KB_train['Summary'] = KB_summary(train_proofs, len(literals), len(premises), KB['Summary']['Number of samples'])
    KB_test['Summary'] = KB_summary(test_proofs, len(literals), len(premises), KB['Summary']['Number of samples'])

    if filename:
        train_name = filename + '_train.json'
        test_name = filename + '_test.json'
        save_data(KB_train, train_name)
        save_data(KB_test, test_name)

        print('The files "{}" and "{}" have been created in "{}"'.format(train_name, test_name, getcwd()))

    return KB_train, KB_test

# MAIN FUNCTION
# to print a function description use 'print(foo.__doc__)'
def create_knowledge_base(filename = 'kb_1', **options):
    '''
    Creates a single knowledge base with at most one path to every node.
    Options:
    - n_levels: (default = 10) the number of levels for the tree
    - range_per_level: (default = (10, 20)) min and max number of nodes for each level
    - probability: (default = 0.5) probability of adding an edge between two nodes
    - used_premises: (default = ['all']) number of used premises for each proof
    - create_files: (default = True) whether or not write files
    - colored_graph: (default = True) whether or not coloring the graph ('create_files' must be True)
    '''
    # CHECK OPTIONS
    # number of levels of the tree
    if 'n_levels' in options.keys():
        L = options['n_levels'] # integer > 1
    else:
        L = 11 # default value

    # range of nodes per level
    if 'range_per_level' in options.keys():
        min_n, max_n = options['range_per_level'] # range: pair of integers
    else:
        min_n, max_n = (3, 6) # default range
    
    # probability of adding an edge between two nodes
    if 'probability' in options.keys():
        P = options['probability'] # float between 0 and 1
    else:
        P = 0.5 # default probability

    # number of used premises for each proof
    if 'used_premises' in options.keys():
        n_prem = options['used_premises'] # list of integers: number of used premises in a proof
    else:
        n_prem = ['all'] # default number (all)

    # create files
    if 'create_files' in options.keys():
        files = options['create_files'] # bool: if 'False', then do not create files
    else:
        files = True # default value

    # # colored graph
    # if 'colored_graph' in options.keys():
    #     w_color = options['colored_graph'] # bool: if 'False', then creates a graph without coloring
    # else:
    #     w_color = True # default value

    # provided graph
    # if 'graph' in options.keys():
    #     graph = options['graph'] # a '.gv' filename
    # else:
    #     graph = None # no graph provided (default)

    # create more than one sample (with literals shuffled)
    if 'samples' in options.keys():
        samples = options['samples'] # integer > 1
    else:
        samples = 1

    # GENERATE DAG
    # if graph provided (grphviz file)
    if graph: 
        E, levels, literals = dag_from_file(graph)
    # generate random graph    
    else: 
        E, levels, literals = random_tree(min_n, max_n, L, P)

    # creates a dictionary 'KB' with all possible proofs from the tree
    KB = {'Literals': sorted(literals)}    

    for s in range(1, samples+1):
        if s > 1:
            E, levels, literals = literals_shuffling(E, levels, literals)

        proofs, sproofs = create_proofs(E, literals, n_prem)
        if proofs:
            premises = [prem(i, t) for i, t in E]                    
            KB['sample {}'.format(s)] = {'Premises': premises, 'Proofs': proofs, 'Sub_proofs': sproofs, 'Graphviz': (E, levels)}
        else:
            return print('No proof can be generated with the currently selected options')

    KB['Summary'] = KB_summary(proofs, len(literals), len(premises), samples)
    # SAVE DICTIONARY AND GRAPH TO FILES
    if files:    
        file_gv = filename + '.gv'     # graphviz file name
        file_json = filename + '.json' # json file name

        save_data(KB, file_json)
        create_dot(E, levels, file_gv)

        print('The files "{}" and "{}" have been created in "{}"'.format(file_json, file_gv, getcwd()))

    return KB 

# DEFINITION OF SYMBOLS
connective = '->'