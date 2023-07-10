import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
file_names = ['norway_bus_1', 'norway_bus_2', 'norway_bus_3', 'norway_bus_4', 'norway_bus_5']

# 모델 학습 파라미터
batch_size = 32
epochs = 10
timestep = 5  # timestep을 5로 설정

# 각 파일에 대해 모델을 학습하고 예측하는 과정
for file_name in file_names:
    file_path = dataset_folder + file_name
    data = load_data(file_path)
    X, y, scaler = preprocess_data(data, timestep)

    # LSTM 모델 구성
    model = build_lstm_model(input_shape=(timestep, 1))  # 입력 차원을 (timestep, 1)로 변경

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    # 예측
    predicted_data_scaled = model.predict(X)
    predicted_data = scaler.inverse_transform(np.concatenate((X[:, -1:], predicted_data_scaled), axis=1))
    predicted_throughput = predicted_data[:, 1]

    # 예측 결과 출력
    print(f'File: {file_name}')
    print('Actual Throughput: ', data[timestep:, 1])  # timestep 이후의 실제 처리량 출력
    print('Predicted Throughput: ', predicted_throughput)
    print('-----------------------------')
