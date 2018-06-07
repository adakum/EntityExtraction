import torch.nn as nn

class RNNModel(nn.Module):

	def __init__(self, embedding_size,
						 bidir,
						 hidden_units,
						 vocab_size,
						 batch_size):

		self.embedding_size = embedding_size
		self.bidir = bidir
		self.hidden_units = hidden_units
		self.vocab_size = vocab_size
		self.batch_size = batch_size

		# init embedding matrix, init the wweight also for this
		# embedding matrix size vocab_size * embeding_size
		self.embedding_matrix = nn.Embedding(self.vocab_size, self.embedding_size)


		# negative log likelhood loss
		self.loss = nn.NLLLoss()
		# rnn
	 	# for now lets got for LSTM
	    # do check how the model performs for GRU  
		self.rnn = nn.LSTM(input_size = self.embedding_size,
						 	hidden_size = self.hidden_units,
						 	num_layers = 1,
						 	bias = True,
						 	batch_first = True,
						 	dropout = 0,
						 	bidirectional = self.bidir)

		
		self.hidden_state_0 = self.init_hidden_state(num_layers, bidir, hidden_size, batch_size)
		
		self.cell_state_0 	= self.init_cell_state(num_layers, bidir, hidden_size, batch_size)

	def init_hidden_state(num_layers, bidir, hidden_size, batch_size ):

		# if bidirectioal, size doubles
		if bidir:
			row_size = num_layers*2
		else:
			row_size = num_layers

		return Variable(torch.zeros([num_layers*bidir, batch_size, hidden_size ]))

	def init_cell_state(num_layers, bidir, hidden_size, batch_size ):

		# if bidirectioal, size doubles
		if bidir:
			row_size = num_layers*2
		else:
			row_size = num_layers

		return Variable(torch.zeros([num_layers*bidir, batch_size, hidden_size]))



	def forward(self, queries, target):
		# queries
			 # contains queries 
			 # dimension : batch_size * lenth
		# target 
			 # dimension : batch_size  

		# steps 
			# 1 apply embedding
			# 2 run rnn 
			# 3 apply softmax
			# 4 compute loss 
		
		# step 1
		embdd_query = self.embedding_matrix(queries)

		# step 2
		output, (h_t, c_t) = self.rnn(embdd_query, self.hidden_state_0, self.cell_state_0)

		# transpose, make batch_size the first axis
		h_t = h_t.transpose(0,1)		 











		

