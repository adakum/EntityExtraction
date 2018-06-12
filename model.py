import torch
import torch.nn as nn
from torch.utils import data

class RNNModel(nn.Module):

	def __init__(self, embedding_size,
						 bidir,
						 hidden_units,
						 vocab_size,
						 batch_size,
						 num_layers,
						 num_entities):
		
		super(RNNModel, self).__init__()

		self.embedding_size = embedding_size
		self.bidir = bidir
		self.hidden_units = hidden_units
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.num_layers = num_layers
		
		# init embedding matrix, init the wweight also for this
		# embedding matrix size vocab_size * embeding_size
		self.embedding_matrix = nn.Embedding(self.vocab_size, self.embedding_size)

			
		# rnn
	 		# for now lets got for LSTM
	    	# do check how the model performs for GRU  
		self.rnn = nn.LSTM(input_size = self.embedding_size,
						 	hidden_size = self.hidden_units,
						 	num_layers = self.num_layers,
						 	bias = True,
						 	batch_first = True,
						 	dropout = 0,
						 	bidirectional = self.bidir)

		self.hidden_state_0 = self.init_hidden_state(self.num_layers, self.bidir, self.hidden_units, self.batch_size)
		
		self.cell_state_0 	= self.init_cell_state(self.num_layers, self.bidir, self.hidden_units, self.batch_size)

		# linear layer 
		if bidir:
			self.layer_1 = nn.Linear(self.hidden_units*2*self.num_layers, num_entities)
		else:
			self.layer_1 = nn.Linear(self.hidden_units*self.num_layers, num_entities)

		# softmax
		self.logSoftmax = nn.LogSoftmax(dim=1) 


	def init_hidden_state(self, num_layers, bidir, hidden_size, batch_size ):

		# if bidirectioal, size doubles
		if bidir:
			row_size = num_layers*2
		else:
			row_size = num_layers

		return torch.zeros([row_size, batch_size, hidden_size ], requires_grad=False)

	def init_cell_state(self, num_layers, bidir, hidden_size, batch_size ):

		# if bidirectioal, size doubles
		if bidir:
			row_size = num_layers*2
		else:
			row_size = num_layers

		return torch.zeros([row_size, batch_size, hidden_size], requires_grad=False)

	def forward(self, queries):
		# queries
			 # contains queries 
			 # dimension : batch_size * lenth
		# target 
			 # dimension : batch_size  

		# steps 
			# 1 apply embedding
			# 2 run rnn 
			# 3 transpose the output hidden_state
			# 4 apply softmax
			# 5 compute loss 
		
		# step 1
		embdd_query = self.embedding_matrix(queries.type(torch.LongTensor))

		# step 2
		rnn_out, (h_t, c_t) = self.rnn(embdd_query, (self.hidden_state_0, self.cell_state_0))

		# transpose, make batch_size the first axis
		h_t = h_t.transpose(0, 1)
		h_t = h_t.view(self.batch_size, -1)

		# h_t
			# dimension : batch_size * vec_size

		# apply linear projection
		nn_out = self.layer_1(h_t)

		# apply softmax
		nn_out = self.logSoftmax(nn_out)

		return nn_out












		
