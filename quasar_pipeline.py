import sys
import json
from sklearn.externals import joblib

from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from Classifier import Classifier
from MultinomialNaiveBayes import MultinomialNaiveBayes
from Evaluator import Evaluator
from TfIdfFeaturizer import TfIdfFeaturizer
from SVMClassifier import SVMClassifier
from mlpClassifier import mlpClassifier

class Pipeline(object):
	def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstances, classifierInstances):
		self.retrievalInstance = retrievalInstance
		self.featurizerInstances = featurizerInstances
		self.classifierInstances = classifierInstances
		trainfile = open(trainFilePath, 'r')
		self.trainData = json.load(trainfile)
		trainfile.close()
		valfile = open(valFilePath, 'r')
		self.valData = json.load(valfile)
		valfile.close()
		self.question_answering()

	def makeXY(self, dataQuestions):
		X = []
		Y = []
		for question in dataQuestions:
			
			long_snippets = self.retrievalInstance.getLongSnippets(question)
			short_snippets = self.retrievalInstance.getShortSnippets(question)
			
			X.append(short_snippets)
			Y.append(question['answers'][0])
			
		return X, Y


	def question_answering(self):
		dataset_type = self.trainData['origin']
		candidate_answers = self.trainData['candidates']
		X_train, Y_train = self.makeXY(self.trainData['questions'][0:1000])
		X_val, Y_val_true = self.makeXY(self.valData['questions'])

                for featurizer in self.featurizerInstances:
                   for classifier in self.classifierInstances:
                        print "Running pipeline with featurizer: ",featurizer , " and classifier ", classifier 
                        #featurization
		        X_features_train, X_features_val = featurizer.getFeatureRepresentation(X_train, X_val)
		        self.clf = classifier.buildClassifier(X_features_train, Y_train)
		
		        #Prediction
		        Y_val_pred = self.clf.predict(X_features_val)
		
                        # Evaluation
    		        self.evaluatorInstance = Evaluator()
		        a =  self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
		        p,r,f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
		        print "Accuracy: " + str(a)
		        print "Precision: " + str(a)
		        print "Recall: " + str(a)
		        print "F-measure: " + str(a)
                        print '\n'		


if __name__ == '__main__':
	trainFilePath = sys.argv[1] #please give the path to your reformatted quasar-s json train file
	valFilePath = sys.argv[2] # provide the path to val file
	retrievalInstance = Retrieval()
	featurizerInstance = [ CountFeaturizer() , TfIdfFeaturizer()]
	classifierInstance = [ MultinomialNaiveBayes() , SVMClassifier(), mlpClassifier()] 
	trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance)
