# POS-Tagging-using-HMM

#### Check CSCI_2023_HW2.pdf for the complete problem description

## Gist of the problem:
- POS Tagging using HMM
- used the Wall Street Journal section of the Penn Treebank to build the HMM model
- Two Algorithms: 
    - Greedy Decoding with HMM
    - Viterbi Decoding with HMM


## Results:

### Greedy Decoding with HMM
- Accuracy on dev data : 0.9331 
- Prediction for test data is generated and stored as in train data. -greedy.out

#### Viterbi Decoding with HMM
- Accuracy on dev data : 0.9475 
- Prediction for test data is generated and stored as in train data. - viterbi.out



## Reproducing the results

- This entire code is in the python file "hw2.py".
- Notebook version is available in the experiment-hw2.ipynb 
- The code generates 4 output files: "vocab.txt", "hmm.json", "greedy.out", and "viterbi.out" (all these are found in the output    directory).

### How to run the code:
-  Ensure that the data folder is in the same directory as the code or specify the correct data path in lines 4, 5 and 6 of the hw2.py file.
-  To run the code, you can either type "python hw2.py" in the command line or open the file in an editor.
-  Once the code is run, the output files will be generated in the same directory as the code.
- Note: viterbi algorithm may take upto 5 minutes to generate the ouput files.
 