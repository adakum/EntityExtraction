import torch.nn as nn

class RNNModel(nn.Module):

	def __init__(self, embedding_size,
						 bidir,
						 hidden_units,
						 vocab_size):

		self.embedding_size = embedding_size
		self.bidir = bidir
		self.hidden_units = hidden_units
		self.vocab_size = vocab_size

		# init embedding matrix, init the wweight also for this
		self.embedding_matrix = nn.Embedding(self.vocab_size, self.embedding_size)

		# rnn 
		self.rnn = nn.LSTM(input_size = self.embedding_size,
						 	hidden_size = self.hidden_units,
						 	num_layers = 1,
						 	bias = True,
						 	batch_first = True,
						 	dropout = 0,
						 	bidirectional = self.bidir)

		

