import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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


# 데이터 전처리 함수
def preprocess_data(data, timestep):
    # 정규화
    scaler_throughput = MinMaxScaler()  # Throughput scaler
    data_scaled = scaler_throughput.fit_transform(data.reshape(-1, 1)).reshape(-1)

    # 입력 데이터와 타깃 데이터 분리
    X, y = [], []
    for i in range(len(data_scaled) - timestep):
        X.append(data_scaled[i:i + timestep])
        y.append(data_scaled[i + timestep])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler_throughput


# LSTM 모델 구성 함수
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dense(32))
    model.add(LSTM(16, return_sequences=True))
    model.add(Dense(8))
    model.add(LSTM(8))
    model.add(Dense(1))
    return model


# 데이터셋 경로
dataset_folder = 'dataset(bandwidth)/'
file_counts = {
    'data(tram merge)': 1        # 교통수단 별로 결과 파일 생성
}

# 모델 학습 파라미터
batch_size = 24
epochs = 10
timestep = 5  # timestep 설정

# 결과 저장 경로
result_folder = 'results(merge bandwidth prediction)/'
os.makedirs(result_folder, exist_ok=True)

# 각 파일에 대해 모델을 학습하고 예측하는 과정
for file_prefix, count in file_counts.items():
    total_loss = 0.0
    total_rmse = 0.0  # 파일별 RMSE 누적 변수
    for n in range(1, count + 1):
        file_name = f"{file_prefix}.txt"
        file_path = os.path.join(dataset_folder, file_name)
        data = load_data(file_path)
        X, y, scaler = preprocess_data(data, timestep)

        # 데이터 분할
        X_train, X_test = X[:1920 - timestep], X[1920 - timestep:2668 - timestep]  # 행 번호로 분할
        y_train, y_test = y[:1920 - timestep], y[1920 - timestep:2668 - timestep]  # 행 번호로 분할

        # LSTM 모델 구성
        model = build_lstm_model(input_shape=(timestep, 1))  # 입력 차원을 (timestep, 1)로 변경

        custom_adam = Adam(learning_rate=0.01,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False,
                           decay=0.0
                           )

        # 모델 컴파일
        model.compile(optimizer=custom_adam, loss='mean_squared_error')

        # 모델 학습
        loss_callback = LossCallback()  # LossCallback 객체 생성
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[loss_callback])

        # 예측
        predicted_data_test_scaled = model.predict(X_test)

        # 그래프 그리기 - 스케일된 데이터 (Train)
        plt.figure(figsize=(14, 5))
        plt.plot(y_train, label='Actual')
        plt.title(f'Scaled values - Train for {file_name}')
        plt.legend()
        plt.show()

        # 그래프 그리기 - 스케일된 데이터 (Test)
        plt.figure(figsize=(14, 5))
        plt.plot(y_test, label='Actual')
        plt.title(f'Scaled values - Test for {file_name}')
        plt.legend()
        plt.show()

        # 그래프 그리기 - 스케일된 데이터 (Test)
        plt.figure(figsize=(14, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(predicted_data_test_scaled, label='Predicted')
        plt.title(f'Scaled values - Test vs Predicted for {file_name}')
        plt.legend()
        plt.show()

        # 예측 결과 저장
        result_file_train_name = f"{file_prefix}_train_result.txt"
        result_file_train_path = os.path.join(result_folder, result_file_train_name)
        with open(result_file_train_path, 'w') as result_file:
            for actual in y_train:
                result_file.write(f"{actual}\n")

        result_file_test_name = f"{file_prefix}_test_result.txt"
        result_file_test_path = os.path.join(result_folder, result_file_test_name)
        with open(result_file_test_path, 'w') as result_file:
            for i in range(len(predicted_data_test_scaled)):
                # timestep을 고려한 실제 데이터와 예측 데이터 작성
                if i < len(y_test):
                    actual = data[i + timestep + 1920]  # 실제 데이터
                    actual_scaled = y_test[i]  # scale된 실제 데이터
                    predicted = scaler.inverse_transform(predicted_data_test_scaled[i].reshape(-1, 1))[0][
                        0]  # inverse scale된 예측 데이터
                    predicted_scaled = predicted_data_test_scaled[i][0]  # scale된 예측 데이터
                    result_file.write(f"{actual}\t{actual_scaled}\t{predicted}\t{predicted_scaled}\n")
                else:
                    # 마지막 구간에 대한 실제 데이터는 없으므로, 0으로 채움
                    actual = 0
                    actual_scaled = 0
                    predicted = scaler.inverse_transform(predicted_data_test_scaled[i].reshape(-1, 1))[0][0]
                    predicted_scaled = predicted_data_test_scaled[i][0]
                    result_file.write(f"{actual}\t{actual_scaled}\t{predicted}\t{predicted_scaled}\n")

        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_test, predicted_data_test_scaled))

        # RMSE 출력
        print(f"RMSE for {file_name}: {rmse}")

        # 누적 RMSE 계산
        total_rmse += rmse

        # 평균 RMSE 출력
    avg_rmse = total_rmse / count
    print(f"Average RMSE for {file_prefix}: {avg_rmse}")

