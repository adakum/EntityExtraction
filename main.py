import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.onnx


parser = argparse.ArgumentParser(description='v.0.1 - Entity and Intent Classification')
#  data arguments
parser.add_argument('--train_data', type=str, default='./data/TrainingData')
parser.add_argument('--test_data', type=str, default='./data/TestData')

# model arguments
parser.add_argument('--embedding_size', type=int, default = 256, help='size of word embeddings')
parser.add_argument('--hidden_units', type=int, default = 128, help='num of hidden units per layer')

parser.add_argument('--bidir', dest='bidir', action='store_true')
parser.add_argument('--no_bidir', dest='bidir', action='store_false')
parser.set_defaults(feature=False)

parser.add_argument('--epochs', type=int, default=2)

parser.add_argument('--seed', type=int, default=108)

args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
