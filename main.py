import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.onnx
import logging
from datautils import textData
from model import RNNModel
from torch.utils import data

parser = argparse.ArgumentParser(description='v.0.1 - Entity and Intent Classification')

# program arguments
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no_train', dest='train', action='store_false')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--no_test', dest='test', action='store_false')
parser.add_argument('--inference', dest='inference', action='store_true')
parser.add_argument('--no_inference', dest='inference', action='store_false')

# logging details
parser.add_argument('--log_file', type = str, default = './logs/log_file')

#  data arguments
parser.add_argument('--train_data', type=str, default='./data/trainingData')
parser.add_argument('--test_data', type=str, default='./data/TestData')


# model arguments
parser.add_argument('--batch_size', type = int, default = 64, help ='batch_size')
parser.add_argument('--embedding_size', type = int, default = 256, help ='size of word embeddings')
parser.add_argument('--hidden_units', type = int, default = 128, help = 'num of hidden units per layer')
parser.add_argument('--bidir', dest='bidir', action ='store_true')
parser.add_argument('--no_bidir', dest='bidir', action ='store_false')
parser.add_argument('--num_layers', type=int, default = 1, help = 'num layers of Encoder')
parser.add_argument('--num_entities', type=int, default = 80000, help = 'num of entities')
parser.add_argument('--num_intent', type=int, default = 80000, help = 'num of intent')
parser.add_argument('--epochs', type = int, default = 1)
parser.add_argument('--vocab_size', type = int, default = 800)

# optimize and loss functions arguments
# parser.add_argument('--optimi)
parser.add_argument('--learning_rate', type = float, default = 0.01)

parser.add_argument('--seed', type=int, default=108)

# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)


def train():
	
	# training file on disk
	train_file = args.train_data
	
	# training data class
	trainData = textData(train_file, args.vocab_size)
	# model 
	model = RNNModel(embedding_size = args.embedding_size,
						bidir = args.bidir,
						hidden_units = args.hidden_units,
						vocab_size = args.vocab_size,
						batch_size = args.batch_size,
						num_layers = args.num_layers,
						num_entities = args.num_entities)


	# create the genereator for the training set and validation set
	params = { 
				'batch_size' : args.batch_size,
				'shuffle'  : True,
				'num_workers': 1
			 }

	train_gen = data.DataLoader(trainData, **params)
	max_epochs = args.epochs

	# loss function and optimizer 
	loss_func = nn.NLLLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
	
	for epoch in range(max_epochs):
		for batch_x, batch_y in train_gen:
			if batch_y.size()[0] < args.batch_size:
				continue
			
			# make zero grad
			optimizer.zero_grad()
			
			output = model(batch_x)
			loss = loss_func(output, batch_y)			
			loss.backward()
			optimizer.step()
			print(loss)		


if __name__ == "__main__":
	print("ck_1 - *** Starting *** ")
	args = parser.parse_args()

	if args.train:
		train()











