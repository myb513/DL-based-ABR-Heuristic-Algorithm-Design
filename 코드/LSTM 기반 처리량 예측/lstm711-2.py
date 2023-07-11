import time

# ...

# 각 파일에 대해 모델을 학습하고 예측하는 과정
for file_prefix, count in file_counts.items():
    total_loss = 0.0
    total_rmse = 0.0  # 파일별 RMSE 누적 변수
    total_time = 0.0  # 파일별 시간 누적 변수
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

        custom_adam = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, decay=0.0)

        # 모델 컴파일
        model.compile(optimizer=custom_adam, loss='mean_squared_error')

        # 모델 학습
        loss_callback = LossCallback()  # LossCallback 객체 생성

        start_time = time.time()  # 시작 시간 측정
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[loss_callback])
        end_time = time.time()  # 종료 시간 측정
        elapsed_time = end_time - start_time  # 경과 시간 계산

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
        total_time += elapsed_time

        print(f"File: {file_name}")
        print(f"Result saved: {result_file_path}")
        print(f"RMSE for {file_name}: {rmse}")
        print(f"Time elapsed for {file_name}: {elapsed_time} seconds")
        print('-----------------------------')

    # 파일별 평균 손실 및 RMSE 출력
    average_loss = total_loss / count
    average_rmse = total_rmse / count
    average_time = total_time / count
    print(f"Average Loss for {file_prefix}: {average_loss}")
    print(f"Average RMSE for {file_prefix}: {average_rmse}")
    print(f"Average Time elapsed for {file_prefix}: {average_time} seconds")
    print('=============================')
