#This will contain the model definitions and the code for building/specifying a particular model
import numpy as np

class Model:
	def __init__(self,
			modelType=NULL,
			utterancePrior=NULL,
			mappings=NULL,
			meanings=NULL,
			utterances=NULL,
			alpha=1.0,
			maxDepth=0):
		self.modelType = modelType
		self.utterancePrior=utterancePrior
		self.mappings=mappings
		self.meanings=meanings
		self.utterances=utterances
		self.alpha=alpha
		self.maxDepth=maxDepth

		#If the mappings are specified as a layer-dependent matrix, set the maxDepth
		#Equal to the number of layers that have been specified. Otherwise, set the max depth equal to infinity

		#Consider making the defaults for paramaters such as the utterance prior uniform

	def iterate(self, numIterations):
		if (numIterations>maxDepth) and (maxDepth!=0):
			raise ValueError("Your iteration request exceeds the max depth of this model")
		if numIterations==1:
			#If the base case, compute the base case
			return __computeBase()
		else:
			#Otherwise compute the higher order case on top of the result from the base case
			return __computeRecursion(iterate(self, numIterations-1))

	def __computeBase():#Might need seperate listener and speaker functions or at least a paramater saying which one is current
		#Compute the base case model predictions

	def __computeRecursion():
		#Compute a recursive step on top of the given model
