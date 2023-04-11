import keras
import keras.layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

def create_processed_data(data, lookback):
    X_data, y_data = [], []
    for i in range(len(data) - lookback):
        X_data.append(data[i:i+lookback])
        y_data.append(data[i+lookback])
    return np.array(X_data), np.array(y_data)


data = pd.read_csv('data/video1_data_t23.txt', delimiter=' ', header=None, names=['time', 'throughput', 'heuristic'])

# 기존 데이터에서 throughput 열만 추출하여 DataFrame 생성
df = pd.DataFrame({'throughput': data['throughput']})

# 처음 20개 구간은 throughput 열의 전체 평균값으로 채움
df['heuristic'] = df['throughput'].rolling(window=10, min_periods=1).mean()

# 새로 계산된 heuristic 열과 실제 throughput 열의 차이를 계산하여 diff 열에 추가
df['diff'] = df['throughput'] - df['heuristic']

#df에 heuristic2 열을 추가하고 data에 있던 열을 그대로 들고옴.
df['heuristic2'] = data['heuristic']

# 이동 평균 계산
rolling_mean = df['throughput'].rolling(window=10, min_periods=1).mean()
rolling_mean = np.roll(rolling_mean, shift=1) # 1칸씩 이동
rolling_mean[0] = 0 # 첫 번째 값은 0으로 설정
df['rolling_mean'] = rolling_mean

# 이동 평균과의 차이 계산
df['diff_rm'] = df['throughput'] - df['rolling_mean']

df.insert(0, 'time', range(len(df)))

print(df.head())

scaler = MinMaxScaler()
df['throughput'] = scaler.fit_transform(data['throughput'].values.reshape(-1,1))

# df를 훈련 데이터와 시험 데이터로 나누기
train_size = int(len(df) * 0.7) # 데이터의 70%를 훈련 데이터로 사용
train_data = df[:train_size]
test_data = df[train_size:]

# 입력 데이터 준비
lookback = 10 # 타임스텝 추 가
input_data = train_data['throughput'].values
X_data, y_data = create_processed_data(input_data, lookback) # 타임스텝 추가

# LSTM 모델 정의
# inputs = keras.Input(shape=(1, 1))
inputs = keras.Input(shape=(lookback, 1)) # 타임스텝 고려
x = keras.layers.LSTM(32)(inputs)
# x = keras.layers.Dense(16)(x)  # Dense 레이어 추가
x = keras.layers.Dense(16)(x)  # Dense 레이어 추가
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

custom_adam = Adam(learning_rate=0.005,
                   beta_1=0.9,  # 일차 모멘텀 추정값에 대한 지수 감쇠율, 일반적으로 0.9 권장
                    beta_2=0.999,  # 이차 모멘텀 추정값에 대한 지수 감쇠율, 일반적으로 0.999 권장
                   epsilon=1e-07,  # 0으로 나누는 것을 방지하기 위한 작은 상수
                   amsgrad=False,  # AMSGrad를 사용할지 여부 (논문에서 제안된 Adam의 변형)
                   decay=0.0
                   )

# 모델 컴파일
model.compile(optimizer=custom_adam, loss='mse')

# 모델 학습
# history = model.fit(x=input_data[:-1, np.newaxis, np.newaxis], y=input_data[1:, np.newaxis], epochs=100, verbose=0)
history = model.fit(x=X_data, y=y_data, epochs=50, verbose=0)

# test_data로부터 heuristic 추출
test_heuristic = test_data['rolling_mean'].values[:-1]



# 시험 데이터를 사용하여 모델 예측
# test_input = test_data['throughput'].values[:-1, np.newaxis, np.newaxis]
test_input = test_data['throughput'].values[:-lookback]

# # 실제 데이
test_target = test_data['throughput'].values[lookback:]

test_X_data, test_y_data = create_processed_data(test_input, lookback)
test_X_data = test_X_data.reshape(test_X_data.shape[0], lookback, 1)

test_predictions = model.predict(test_X_data)

# 예측값의 개수 맞춰주기
test_predictions = np.concatenate([test_predictions, np.zeros((lookback, 1))])

# heuristic 방법을 통한 예측값
# test_avg = test_data['rolling_mean'].values[1:-1, np.newaxis, np.newaxis]
# test_avg = test_data['rolling_mean'].values[lookback:-1, np.newaxis, np.newaxis]
test_avg = test_data['heuristic2'].values[lookback:-1, np.newaxis, np.newaxis]
test_avg = test_avg[:-lookback+1]

# txt 파일로 저장
with open('data/v1heuristic_data_494949.txt', 'w') as f:
    for i in range(len(test_avg)):
        f.write("{} {:.6f}\n".format(i, test_avg[i, 0, 0]))


# 역변환 수행
test_target = scaler.inverse_transform(test_target.reshape(-1, 1)) #test data 실제 값
test_predictions = scaler.inverse_transform(test_predictions) #예측 값

test_target = test_target[:-lookback]
test_predictions = test_predictions[:-lookback]
# 예측값 저장하기
with open("data/v1LSTM_data_494949.txt", 'w') as f:
    for i in range(len(test_predictions)):
        f.write(f"{i} {test_predictions[i][0]:.6f}\n")

    # 예측값 저장하기
with open("data/v1Actual_data_494949.txt", 'w') as f:
    for i in range(len(test_target)):
        f.write(f"{i} {test_target[i][0]:.6f}\n")



# diff_pred = test_target - test_predictions
# actual과 prediction의 차이
diff_pred = test_target - test_predictions[:len(test_target)]
# actual과 avg의 차이
diff_avg = test_target - test_avg
mse_pred = np.mean(np.square(diff_pred))
mse_avg = np.mean(np.square(diff_avg))

print('actual:', test_target[40:60])
print('prediction:', test_predictions[40:60])
print('avg:', test_avg[40:60, 0, 0])


# 시각화 코드

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 첫번째 축에 그래프 그리기
ax1.plot(test_target, label='actual', alpha=0.7)
ax1.plot(test_predictions, label='prediction', alpha=0.7)
ax1.plot(test_avg[:, 0,0], label='avg', alpha=0.7)
ax1.set_xlabel('Time')
ax1.set_ylabel('Throughput')
ax1.legend()

# 두번째 축에 그래프 그리기
ax2.plot(diff_pred, label='actual-prediction', alpha=0.7)
ax2.plot(diff_avg[:, 0,0], label='actual-avg', alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Difference')
ax2.legend()
plt.text(0.5, 0.9, f"MSE(DL based): {mse_pred:.4f}", transform=plt.gca().transAxes)
plt.text(0.5, 0.85, f"MSE(heuristic): {mse_avg:.4f}", transform=plt.gca().transAxes)

plt.show()
