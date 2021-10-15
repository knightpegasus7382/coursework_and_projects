************ FOR THE HMM QUESTION **********************

In order to run the HMM code hmmkmeans.py and evaluate it:

1. Run 
	python3 hmmkmeans.py 
   to get the K-means cluster indices for vectors for each digit, in the files train_<digit>.hmm.seq.

2. Run 
	./train_hmm train_<digit>.hmm.seq 1234 <number of states> <number of symbols=16> .01 
   to train the respective HMMs and obtain the parameters in the files train_<digit>.hmm.seq.hmm.


3. Run 
	./test_hmm test_all.hmm.seq train_<digit>.hmm.seq.hmm > alphas_<digit>.txt
   to store the output of alphas (log likelihoods) for each sequence into .txt output files, in order to run hmm_results_extract.py

4. Run 
	python3 hmm_results_extract.py 
   in order to obtain the predicted digit labels of test data, and the accuracy percentage.

	
************ FOR THE CONNECTED DIGITS QUESTION **********************

1. Running 
	python3 gen_conn_labels.py 
   will generate the label data for the given dev directory test files for the connected-digits question. It will generate them in the dev directory itself, and write them into the file "truthlabels.txt". (IT HAS ALREADY BEEN RUN IN THE SUBMISSION, so the next step can be performed directly.)

2. Run 
	python3 connected.py 
   to give the Top 3 Predictions for either any of the "dev" directory test files. (This section of the code is COMMENTED in the submission. It will have to be uncommented to get this functionality.)
   Or the Top 3 Predictions for all the blind-test files. (This section of the code is already UNCOMMENTED, so running the code directly prints the required blind-test output in the console.)


