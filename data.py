import os,sys

class textData():
	def __init__(self, filename, vocab_size):
		
		# filename would be in this format for now 
			# query	entity 
		
		# right now store tuples of query and entity 
		self.data = [] 
		self.queries = []
		self.target = []

		# dictionary word to index
		self.word2idx = {}

		# dictionary index to word
		self.idx2word = {}

		# entity to index
		self.ent2idx = {}

		# index to entity
		self.idx2ent = {}

		# vocab size
		self.vocab_size = vocab_size
		
		with open(filename, "r") as reader:
			self.data = [(line.strip().split("\t")[0], line.strip().split("\t")[1]) for line in reader.readlines()]

		# get index for each word
		self.Word2Index()

		# convert input queries into seq of idxs
		self.Queries2Idx()

	def getSize(self):
		return len(target)

	def getElement(self, i):
		return 0

	def Queries2Idx(self):
		
		for (query, entity) in self.data:
			inp_seq = [self.word2idx[x] for x in query.split()]
			self.queries = queries + [inp_seq]
			self.target  = target + [self.ent2idx[entity]]



	def Word2Index(self):
		wordcnt = {}
		
		queries = data.keys()
		entity = data.values()

		for query in queries:
			for w in query.split():
				try:
					word2cnt[w] = word2cnt[w] + 1
				except:
					word2cnt[w] = 1 

		
		words = [k for k in sorted(wordcnt, key=wordcnt.get, reverse=True)]

		# take top self.vocab_size words
		if len(words) > self.vocab_size :
			words = words[:self.vocab_size]


		for i,w in enumerate(words):
			self.word2idx[word] = i
			self.idx2word[i] = word

		entity = list(data.values())
		for i,n in enumerate(entity):
			self.ent2idx[n] = i
			self.idx2ent[i] = n












