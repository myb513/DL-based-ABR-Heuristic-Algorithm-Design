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
    model.add(Dense(32))
    model.add(LSTM(16, return_sequences=True))
    model.add(Dense(8))
    model.add(LSTM(8))
    model.add(Dense(1))
    return model

# 데이터셋 경로
train_dataset_folder = 'dataset(bandwidth)'
test_dataset_folder = 'dataset(bandwidth test)'

train_file_counts = {
    'norway_bus': 17,
    'norway_car': 9,
    'norway_ferry': 14,
    'norway_metro': 7,
    'norway_train': 15,
    'norway_tram': 40
}

test_file_counts = {
    'norway_bus': 6,
    'norway_car': 3,
    'norway_ferry': 6,
    'norway_metro': 3,
    'norway_train': 6,
    'norway_tram': 16
}

# 모델 학습 파라미터
batch_size = 32
epochs = 400
timestep = 10  # timestep을 10로 설정

# 결과 저장 경로
result_folder = 'results(bandwidth prediction)/'
os.makedirs(result_folder, exist_ok=True)

# 파일별 RMSE 및 평균 RMSE 저장 변수
file_rmse = {}
average_rmse = {}

# 훈련 데이터에 대해 모델을 학습하는 과정
for data_type, count in train_file_counts.items():
    total_loss = 0.0
    for n in range(1, count + 1):
        file_name = f"bandwidth data({data_type}_{n}).txt"
        file_path = os.path.join(train_dataset_folder, file_name)
        data = load_data(file_path)
        X, y, scaler = preprocess_data(data, timestep)

        # LSTM 모델 구성
        model = build_lstm_model(input_shape=(timestep, 1))  # 입력 차원을 (timestep, 1)로 변경

        custom_adam = Adam(learning_rate=0.005,
                           beta_1=0.9,  # 일차 모멘텀 추정값에 대한 지수 감쇠율, 일반적으로 0.9 권장
                           beta_2=0.999,  # 이차 모멘텀 추정값에 대한 지수 감쇠율, 일반적으로 0.999 권장
                           epsilon=1e-07,  # 0으로 나누는 것을 방지하기 위한 작은 상수
                           amsgrad=False,  # AMSGrad를 사용할지 여부 (논문에서 제안된 Adam의 변형)
                           decay=0.0
                           )

        # 모델 컴파일
        model.compile(optimizer=custom_adam, loss='mean_squared_error')

        # 모델 학습
        model.fit(X, y, batch_size=batch_size, epochs=epochs)

        print(f"Train File: {file_name}")
        print('-----------------------------')

        loss = model.evaluate(X, y)
        total_loss += loss

    print(f"Average Loss for {data_type}: {total_loss / count}")
    print('=============================')

# 테스트 데이터에 대해 예측하는 과정
for data_type, count in test_file_counts.items():
    total_rmse = 0.0  # 파일별 RMSE 누적 변수
    for n in range(1, count + 1):
        file_name = f"bandwidth testdata({data_type}_{n}).txt"
        file_path = os.path.join(test_dataset_folder, file_name)
        data = load_data(file_path)
        X, y, scaler = preprocess_data(data, timestep)

        # 예측
        predicted_data_scaled = model.predict(X)
        predicted_data = scaler.inverse_transform(np.concatenate((X[:, -1:], predicted_data_scaled), axis=1))
        predicted_bandwidth = predicted_data[:, 1]

        # 전체 RMSE 계산
        rmse = np.sqrt(mean_squared_error(y, predicted_bandwidth))
        total_rmse += rmse

        print(f"Test File: {file_name}")
        print(f"RMSE for {file_name}: {rmse}")
        print('-----------------------------')

        # 예측 결과 저장
        result_file_name = f"{data_type}_{n}_prediction.txt"
        result_file_path = os.path.join(result_folder, result_file_name)
        with open(result_file_path, 'w') as result_file:
            result_file.write("time_stamp\tPredicted Throughput\n")
            for time, predicted in zip(data[timestep:, 0], predicted_bandwidth):
                result_file.write(f"{time}\t{predicted}\n")

        print(f"Test File Prediction Saved: {result_file_path}")
        print('-----------------------------')

    # 파일별 RMSE 저장
    file_rmse[data_type] = total_rmse / count

    print(f"Average RMSE for {data_type}: {total_rmse / count}")
    print('=============================')

# 평균 RMSE 시각화
data_types = list(file_rmse.keys())
rmse_values = list(file_rmse.values())

plt.bar(data_types, rmse_values)
plt.xlabel('Data Type')
plt.ylabel('RMSE')
plt.title('Average RMSE per Data Type')
plt.xticks(rotation=45)
plt.show()
