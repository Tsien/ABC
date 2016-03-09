def get_neural_net_configuration():
	nn_params = {}
	#Number of hidden dimensions.
	#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
	nn_params['hidden_dimension_size'] = 1024

	#The type of optimizer
	nn_params['opt'] = 'SGD'
        #nn_params['opt'] = 'SGD'
	#nn_params['opt'] = 'Adagrad'

	#The lenth of generated ABC notation
	nn_params['len_gen'] = 500

	#Maximum number of iterations for training
	nn_params['max_iter'] = 50 

	# cut the text in semi-redundant sequences of maxlen characters
	# maxlen is the length of every input sequence
	# step is the interval between two continuous inputs
	nn_params['maxlen'] = 100
	nn_params['step'] = 3

	#The dataset filename 
	nn_params['dataset'] = '../../datasets/jigs.txt'

	#Seed Sentence filename
	nn_params['seed'] = '../../datasets/seedsentence.txt'

	#if you use shell script, do NOT change the following three parameters!
	#The filename of ouput file
	nn_params['generatefile'] = 'generated.txt'

	#The filename of weights file
	nn_params['weightfile'] = 'weights.txt'
	
	#The filename of ouput file
	nn_params['lossfile'] = 'loss.csv'
	return nn_params

