import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

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
    model.add(LSTM(64, input_shape=input_shape))
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
timestep = 5  # timestep을 5로 설정
test_ratio = 0.3  # 테스트 데이터 비율 (train: 0.7, test: 0.3)

# 결과 저장 경로
result_folder = 'results/'
os.makedirs(result_folder, exist_ok=True)

# 각 파일에 대해 모델을 학습하고 예측하는 과정
for file_prefix, count in file_counts.items():
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
        model = build_lstm_model(input_shape=(timestep, 1))  # 입력 차원을 (timestep, 1)로 변경

        # 모델 컴파일
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 모델 학습
        loss_callback = LossCallback()  # LossCallback 객체 생성
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[loss_callback])

        # 예측
        predicted_data_scaled = model.predict(X_test)
        predicted_data = scaler.inverse_transform(np.concatenate((X_test[:, -1:], predicted_data_scaled), axis=1))
        predicted_throughput = predicted_data[:, 1]

        # 예측 결과 저장
        result_file_name = f"{file_prefix}_{n}_result.txt"
        result_file_path = os.path.join(result_folder, result_file_name)
        with open(result_file_path, 'w') as result_file:
            result_file.write("time_stamp\tActual Throughput\tPredicted Throughput\n")
            for time, actual, predicted in zip(time_stamp_test, y_test, predicted_throughput):
                result_file.write(f"{time}\t{actual}\t{predicted}\n")

        # 전체 손실 및 RMSE 누적
        total_loss += loss_callback.total_loss
        rmse = np.sqrt(mean_squared_error(y_test, predicted_throughput))
        total_rmse += rmse

        print(f"File: {file_name}")
        print(f"Result saved: {result_file_path}")
        print(f"RMSE for {file_name}: {rmse}")
        print('-----------------------------')

    # 파일별 평균 손실 및 RMSE 출력
    average_loss = total_loss / count
    average_rmse = total_rmse / count
    print(f"Average Loss for {file_prefix}: {average_loss}")
    print(f"Average RMSE for {file_prefix}: {average_rmse}")
    print('=============================')
