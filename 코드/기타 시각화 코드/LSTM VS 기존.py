import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.total_loss = 0.0

    def on_epoch_end(self, epoch, logs=None):
        self.total_loss += logs['loss']

# 데이터를 로드하는 함수
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            throughput = float(line[1])  # throughput 값만 사용
            data.append(throughput)
    return np.array(data)


def preprocess_data(data, timestep):
    scaler_throughput = MinMaxScaler()
    data_scaled = scaler_throughput.fit_transform(data.reshape(-1, 1)).reshape(-1)


    X, y = [], []
    for i in range(len(data_scaled) - timestep):
        X.append(data_scaled[i:i+timestep])
        y.append(data_scaled[i+timestep])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler_throughput

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    return model

dataset_folder = 'dataset(bandwidth)/'
file_counts = {
    'data(ferry merge)' : 1
}

batch_size = 24
epochs = 10
timestep = 5
test_ratio = 0.3

result_folder = 'results(merge bandwidth prediction)/'
os.makedirs(result_folder, exist_ok=True)

for file_prefix, count in file_counts.items():
    total_loss = 0.0
    total_rmse = 0.0
    for n in range(1, count + 1):
        file_name = f"{file_prefix}.txt"
        file_path = os.path.join(dataset_folder, file_name)
        data = load_data(file_path)
        X, y, scaler = preprocess_data(data, timestep)

        test_size = int(len(X) * test_ratio)
        print(test_size)
        train_size = int(len(X) * (1-test_ratio))

        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        print(len(X_train), len(X_test))
        print(len(y_train), len(y_test))
        # X_train, X_test = X[:train_size - timestep], X[train_size - timestep : len(X)-timestep]
        # y_train, y_test = y[:train_size - timestep], y[train_size - timestep : len(X)-timestep]
        model = build_lstm_model(input_shape=(timestep, 1))

        custom_adam = Adam(learning_rate=0.01,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False,
                           decay=0.0
                           )

        model.compile(optimizer=custom_adam, loss='mean_squared_error')

        loss_callback = LossCallback()

        start_time = time.time()

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[loss_callback])

        predicted_data_scaled = model.predict(X_test)

        end_time = time.time()
        training_prediction_time = end_time - start_time
        print(f'training & Prediction time for {file_name}: {training_prediction_time} seconds')

        predicted_throughput = scaler.inverse_transform(predicted_data_scaled)

        plt.figure(figsize=(14, 5))
        plt.plot(y_train, label='Actual')
        plt.title(f'Scaled values - Train for {file_name}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(14, 5))
        plt.plot(y_test, label='Actual')
        plt.title(f'Scaled values - Test for {file_name}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(14, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(predicted_data_scaled, label='Predicted')
        plt.title(f'Scaled values - Test vs Predicted for {file_name}')
        plt.legend()
        plt.show()

        actual_throughput_unscaled = scaler.inverse_transform(y_test.reshape(-1,1))

        result_file_name = f"{file_prefix}_merge_result_timestep_5.txt"
        result_file_path = os.path.join(result_folder, result_file_name)
        with open(result_file_path, 'w') as result_file:
            for actual, predicted in zip(actual_throughput_unscaled, predicted_throughput):
                result_file.write(f"{actual[0]}\t{predicted[0]}\n")
        # with open(result_file_path, 'w') as result_file:
        #     for i in range(len(predicted_throughput)):
        #         # timestep을 고려한 실제 데이터와 예측 데이터 작성
        #         # if i < len(y_test):
        #         if i < len(y_test):
        #             actual = data[i + timestep + train_size]  # 실제 데이터
        #             predicted = scaler.inverse_transform(predicted_throughput[i].reshape(-1, 1))[0][
        #                 0]  # inverse scale된 예측 데이터
        #             result_file.write(f"{actual}\t{predicted}\n")
        #         else:
        #             # 마지막 구간에 대한 실제 데이터는 없으므로, 0으로 채움
        #             actual = 0
        #             predicted = scaler.inverse_transform(predicted_throughput[i].reshape(-1, 1))[0][0]
        #             result_file.write(f"{actual}\t{predicted}\n")

        rmse = np.sqrt(mean_squared_error(y_test, predicted_data_scaled))
        total_rmse += rmse

        total_loss += loss_callback.total_loss

    avg_rmse = total_rmse / count
    avg_loss = total_loss / (count * epochs)
    print(f'Average RMSE for {file_prefix}: {avg_rmse}')
    print(f'Average loss for {file_prefix}: {avg_loss}')
