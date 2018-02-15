from Classifier import Classifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense

#This is a subclass that extends the abstract class Classifier.
class MLPClassifier(Classifier):

	#The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
	def buildClassifier(self, X_features, Y_train):
                model = Sequential()
                hidden = 128
                optimizer = 'sgd'
                epochs = 20
                out = 1000
                 
		clf = MLPClassifier().fit(X_features, Y_train)
		return clf
