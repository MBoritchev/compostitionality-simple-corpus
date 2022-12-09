# Summary

This is the code accompanying the paper _Compositionality in a simple corpus_, presented at the LIFT TAL 2022 : Journées Jointes des Groupements de Recherche « Linguistique Informatique, Formelle et de Terrain » et « Traitement Automatique des Langues ». The paper can be found here: https://hal.archives-ouvertes.fr/hal-03859310/document#page=64.

This code requires the following specifications to be ran:
- python version 3.7.15
- tensorflow version 2.9.2
- keras version 2.9.0
- sklearn version 1.0.2
- json version 2.0.9
- colour version 0.1.5

Once you have cloned our repository, you can execute the code locally. Please cite the paper if you want to use the resources in this repository. If you have any questions, feel free to ask them: m.vargas-guzman@uw.edu.pl, mboritchev@impan.pl, mmalicki@impan.pl, jakub.szymanik@gmail.com.


# Structure of the repository

This repository contains 4 folders. 
- `Knowledge-base` contains the code and data relative to the Knowledge base: the scripts needed to build it and the ones needed to encode it, along with a built and encoded example.
- `Initial-experiment` contains the code and data relative to the initial experiment in our paper, using the example knowledge base from the `Knowledge-base` folder: the scripts needed to run it along with output logs of our run. There are two options to run this code: (1) open the file "initial_experiment.ipynb" in Google Colab and run each cell; (2) download the repository and open the file "initial_experiment.ipynb" and run it from cell 3.
- `Compositionality-experiments` contains the code and data relative to the compositionality experiments in our paper, using the example knowledge base from the `Knowledge-base` folder: the scripts needed to run them along with output logs of our runs.
- `Hamming-distances` contains the code and data relative to the error analysis of logs from the `Initial-experiment` and `Compositionality-experiments` folders: the scripts needed to run them along with plotted data from our runs.
