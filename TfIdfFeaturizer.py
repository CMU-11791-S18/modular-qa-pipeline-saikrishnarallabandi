from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer


#This is a subclass that extends the abstract class Featurizer.
class TfIdfFeaturizer(Featurizer):

	#The abstract method from the base class is implemeted here to return count features
	def getFeatureRepresentation(self, X_train, X_val):
		tfidf_vect = TfidfVectorizer()
		X_train_counts = tfidf_vect.fit_transform(X_train)
		X_val_counts = tfidf_vect.transform(X_val)
		return X_train_counts, X_val_counts
