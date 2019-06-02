import pickle
from sklearn.externals import joblib

# load pickle file
def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		file_pickle = pickle.load(f)
	return file_pickle


# save pickle file
def save_pickle(data, file_path):
	with open(file_path, 'wb') as f:
		pickle.dump(data, f)