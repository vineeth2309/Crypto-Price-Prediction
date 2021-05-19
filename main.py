#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# %matplotlib inline

class Data:

	def __init__(self, data_path, time_horizon, epochs, batch_size, val_size, load_model = False):
		self.load_model = load_model
		plt.figure(figsize=(16,8))
		self.df = pd.read_csv(data_path)
		self.df.drop(columns=['unix', 'symbol'],axis=1,inplace=True)
		self.scaler = MinMaxScaler(feature_range=(0, 1))

		# print(self.df.head())
		self.df.index = self.df['date']
		self.df.set_index('date', drop=True, append=False, inplace=True, verify_integrity=False)
		self.df = self.df.sort_index()
		self.time_horizon = time_horizon # Total time to look back including current day
		self.epochs, self.batch_size = epochs, batch_size
		self.val_size = val_size
		self.data_prep()
		# self.kmeans()
		self.model_create()
		# self.train()
		self.test()
		# self.plot_data()

	def model_create(self):
		if self.load_model == True:
			self.model = keras.models.load_model('train.h5')
		else:
			self.model = Sequential()
			self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.time_horizon, 7)))
			self.model.add(LSTM(units=50))
			self.model.add(Dense(1))
			self.model.compile(loss='mean_squared_error', optimizer='adam')
		
	def train(self):
		self.model.fit(self.X_train, self.Y_train, epochs = self.epochs, batch_size = self.batch_size, verbose=2)
		self.model.save("train.h5")
		closing_prices = np.zeros((self.Y_train.shape[0] - self.val_size, 1))
		output = self.model.predict(self.X_train)
		closing_prices = np.concatenate((closing_prices, output), axis = 0)
		plt.plot(closing_prices, label='pred', color = "green")

	def test(self):
		output = self.model.predict(self.X_val)
		print(output[0], self.Y_val[0])
		plt.plot(output, label='pred', color = "green")
		plt.plot(self.Y_val, label='Close Price history', color = "red")
		plt.show()

	def kmeans(self):
		self.k_means_data = self.scaled_train_data[:-200,:]
		kmeans = KMeans(init="random", n_clusters=10, n_init=10, max_iter=300)
		kmeans.fit(self.k_means_data)
		print(kmeans.cluster_centers_)

	def data_prep(self):
		# Input -> Past 20 days [Open, High, Low, Close, Adj Close, Volume], Shape = (batch_size, time_horizon, 6)
		# Output -> Predicted Closing Price for the next day, Shape = (batch_size, 1)
		# self.base_data = self.df.values[:500,:]
		# self.base_data = self.df.values[:500,:]
		self.base_data = self.df.values
		# print(self.base_data.shape)
		# print(self.base_data[0])
		self.base_data = self.base_data[~np.isnan(self.base_data.astype(np.float32)).any(axis=1)].astype(np.float32)
		self.scaled_base_data = self.scaler.fit_transform(self.base_data)
		self.X_base, self.Y_base = [], []
		for i in range(self.time_horizon - 1, self.scaled_base_data.shape[0] - 1):
			self.X_base.append(self.scaled_base_data[i - self.time_horizon + 1: i + 1])
			self.Y_base.append(self.scaled_base_data[i + 1][3])
		self.X_base = np.array(self.X_base)
		self.Y_base = np.expand_dims(np.array(self.Y_base), axis = 1)
		self.X_train, self.Y_train = self.X_base[:-self.val_size, :], self.Y_base[:-self.val_size]
		self.X_val, self.Y_val= self.X_base[-self.val_size:, :], self.Y_base[-self.val_size:]
		# print(self.X_train[0], self.Y_train[0])
		# print(self.X_train[1])
		# print(self.X_val.shape, self.Y_val.shape)
		

	def plot_data(self):
		plt.plot(self.Y_train, label='Close Price history', color = "red")
		plt.show()
	
		

if __name__ == "__main__":
	# obj = Data('/home/vineeth_s_subramanyan/Desktop/Binance_ETHUSDT_minute.csv', 20, 20, 4, 20)
	obj = Data('Binance_ETHUSDT_minute.csv', 20, 2, 32, 1000, True)