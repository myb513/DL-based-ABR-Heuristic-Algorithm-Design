import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from keras.optimizers import Adam

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
            time_stamp = float(line[0])
            throughput = float(line[1])
            data.append([time_stamp, throughput])
    return np.array(data)

# 데이터 전처리 함수
def preprocess_data(data, timestep):
    # 정규화
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 입력 데이터와 타깃 데이터 분리
    X, y = [], []
    for i in range(len(data_scaled) - timestep):
        X.append(data_scaled[i:i+timestep, 0])
        y.append(data_scaled[i+timestep, 1])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# LSTM 모델 구성 함수
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    return model

def build_lstm_model_2(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    return model

def build_lstm_model_3(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    return model

# 데이터셋 경로
dataset_folder = 'dataset/'
file_counts = {
    'norway_bus': 23,
    'norway_car': 12,
    'norway_ferry': 20,
    'norway_metro': 10,
    'norway_train': 21,
    'norway_tram': 56
}

# 모델 학습 파라미터
batch_size = 32
epochs = 10
timestep = 10  # timestep을 10로 설정
test_ratio = 0.3  # 테스트 데이터 비율 (train: 0.7, test: 0.3)

# 결과 저장 경로
result_folder = 'results/'
os.makedirs(result_folder, exist_ok=True)

models = [build_lstm_model, build_lstm_model_2, build_lstm_model_3]
model_names = ['Model 1', 'Model 2', 'Model 3']

# 각 모델별 RMSE와 Loss 값을 저장할 리스트
rmse_values = []
loss_values = []

# 각 파일에 대해 모델을 학습하고 예측하는 과정
for file_prefix, count in file_counts.items():
    for model_func, model_name in zip(models, model_names):
        total_loss = 0.0
        total_rmse = 0.0  # 파일별 RMSE 누적 변수
        for n in range(1, count + 1):
            file_name = f"{file_prefix}_{n}"
            file_path = os.path.join(dataset_folder, file_name)
            data = load_data(file_path)
            X, y, scaler = preprocess_data(data, timestep)

            # 데이터 분할
            test_size = int(len(X) * test_ratio)
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]
            time_stamp_test = data[timestep:][len(data) - test_size:, 0]

            # LSTM 모델 구성
            model = model_func(input_shape=(timestep, 1))

            custom_adam = Adam(learning_rate=0.005,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               decay=0.0
                               )

            # 모델 컴파일
            model.compile(optimizer=custom_adam, loss='mean_squared_error')

            # 모델 학습
            loss_callback = LossCallback()
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[loss_callback])

            # 예측
            predicted_data_scaled = model.predict(X_test)
            predicted_data = scaler.inverse_transform(np.concatenate((X_test[:, -1:], predicted_data_scaled), axis=1))
            predicted_throughput = predicted_data[:, 1]

            # 전체 손실 및 RMSE 누적
            total_loss += loss_callback.total_loss
            rmse = np.sqrt(mean_squared_error(y_test, predicted_throughput))
            total_rmse += rmse

        # 파일별 평균 손실 및 RMSE 출력
        average_loss = total_loss / count
        average_rmse = total_rmse / count

        print(f"Model: {model_name}")
        print(f"Average Loss for {file_prefix}: {average_loss}")
        print(f"Average RMSE for {file_prefix}: {average_rmse}")
        print('=============================')

        # RMSE와 Loss 값을 리스트에 저장
        rmse_values.append(average_rmse)
        loss_values.append(average_loss)

# RMSE와 Loss 값을 그래프로 표시
plt.figure(figsize=(10, 6))
plt.plot(model_names, rmse_values, marker='o', label='RMSE')
plt.plot(model_names, loss_values, marker='o', label='Loss')
plt.xlabel('Model')
plt.ylabel('Value')
plt.title('RMSE and Loss for Each Model')
plt.legend()
plt.show()
