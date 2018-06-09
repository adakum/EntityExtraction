import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.onnx
from data import textData

parser = argparse.ArgumentParser(description='v.0.1 - Entity and Intent Classification')

# program arguments
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no_train', dest='train', action='store_false')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--no_test', dest='test', action='store_false')
parser.add_argument('--inference', dest='inference', action='store_true')
parser.add_argument('--no_inference', dest='inference', action='store_false')

#  data arguments
parser.add_argument('--train_data', type=str, default='./data/TrainingData')
parser.add_argument('--test_data', type=str, default='./data/TestData')

# model arguments
parser.add_argument('--embedding_size', type=int, default = 256, help='size of word embeddings')
parser.add_argument('--hidden_units', type=int, default = 128, help='num of hidden units per layer')
parser.add_argument('--bidir', dest='bidir', action='store_true')
parser.add_argument('--no_bidir', dest='bidir', action='store_false')
parser.add_argument('--num_layers', type=int, default = 1, help = 'num layers of Encoder')
parser.add_argument('--num_entities', type=int, default = 80000, help = 'num of entities')
parser.add_argument('--num_intent', type=int, default = 80000, help = 'num of intent')
parser.add_argument('--epochs', type=int, default=2)

parser.add_argument('--seed', type=int, default=108)



# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)

def inference():

def train():

if __name__ == "__main__":
	print("ck_1 - *** Starting *** ")

	args = parser.parse_args()

	if args.train:
		train()











