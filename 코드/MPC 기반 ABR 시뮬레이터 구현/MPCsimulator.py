import itertools
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.optimize import brute

class MPC:
    def __init__(self, buffer_size, bitrates, segment_lengths, target_buffer=60, initial_quality=0):
        self.buffer_size = buffer_size
        self.bitrates = bitrates
        self.segment_lengths = segment_lengths
        self.target_buffer = target_buffer  # target buffer : 유지 되는 목표 버퍼의 크기 , 미구현
        self.current_buffer = 0
        self.current_quality = initial_quality

    def predict(self, throughput_history):

        def objective(x):
            quality_indices = np.asarray(x, dtype=int)
            bitrate_diffs = np.diff([self.bitrates[i] for i in quality_indices])
            bitrate_changes = np.multiply(bitrate_diffs, self.segment_lengths[:len(bitrate_diffs)])

            buffer_changes = np.cumsum(self.segment_lengths)[:-1] - self.current_buffer
            score = np.sum(np.maximum(throughput_history[:-1] - bitrate_changes / buffer_changes, 0) ** 2)
            # print(score)
            return score


        def is_constraint_satisfied(x):
            buffer_history = np.cumsum(self.segment_lengths[:len(x)] * (np.array(x) - self.current_quality))
            return np.all(self.buffer_size >= buffer_history)

        quality_levels = list(range(len(self.bitrates)))


        valid_combinations = []
        for combo in itertools.product(quality_levels, repeat=len(throughput_history)):
            if is_constraint_satisfied(combo):
                valid_combinations.append(combo)
        # print("valid_combinations",valid_combinations)

        scores = [objective(np.ravel(combo)) for combo in valid_combinations]

        best_combination = valid_combinations[np.argmin(scores)]
        print("best",best_combination)

        self.current_buffer = np.cumsum(np.multiply(self.segment_lengths, np.subtract(best_combination, self.current_quality)))[-1]

        self.current_quality = best_combination

        return self.current_quality


throughput_data = []
with open('data/v1heuristic_data_494949.txt', 'r') as file:
    for line in file:
        _, value = line.strip().split()
        throughput_data.append(float(value) * 100)  # Multiply by 100 to convert from Mbps to kbps

# throughput_data = throughput_data[len(throughput_data)-500 : -200]

# print(throughput_data[-1])
print(len(throughput_data))

buffer_size = 400
# bitrates = [200, 412, 616, 821, 1535, 2554, 4037] # kbps of redbull(video #3)
bitrates = [254, 507, 759, 1013, 1254, 1883, 3134, 4952, 9914, 14931]
# bitrates = [490, 979, 1963, 2957, 4934] # kbps of underwater(video #2)
# assumption
# bitrates = [500, 4000, 6000, 7000, 7500]
# segment_lengths = [4] * len(throughput_data)  # Assuming each segment length is 4 seconds
window_size = 5
segment_lengths = [4] * window_size  # Assuming each segment length is 4 seconds
abr = MPC(buffer_size, bitrates, segment_lengths)
qualities = []
buffers = []

for i in range(len(throughput_data) - window_size + 1):
    if(i%window_size == 0):
        window_data = throughput_data[i:i + window_size]
        quality = abr.predict(window_data)
        buffer = abr.current_buffer

        for q in quality:
            qualities.append(q)
        buffers.append(buffer)
        print("{} 구간 계산 완료. ({}/{})".format(i, i, len(throughput_data) - window_size + 1))

# print(qualities)
#
# plt.plot(buffers)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Buffer (seconds)')
# plt.title('Quantitative change of Buffer  over time')
# plt.show()

# 각 시점마다 버퍼 상태가 음수인 경우 카운트
negative_count = sum(1 for buffer_state in buffers if buffer_state < 0)

# 그래프에 추가
plt.plot(buffers)
plt.xlabel('Time (seconds)')
plt.ylabel('Buffer (seconds)')
plt.title('Video 1, Heuristic, Quantitative change of Buffer over time ({} negative)'.format(negative_count))
plt.text(0, -100, '{} negative'.format(negative_count), ha='left', va='top')
plt.show()

print(buffers)

# plt.plot(qualities,'.')
# plt.xlabel('Time (every 4 seconds)')
# plt.ylabel('Quality (index)')
# plt.title('Quality over time')
# plt.show()
# 각 화질이 몇 번 등장했는지 계산
quality_counts = collections.Counter(qualities)

# 그래프에 추가
fig, ax = plt.subplots()
ax.plot(qualities, '.')

# 각 화질이 몇 번 등장했는지 텍스트로 추가
for quality, count in quality_counts.items():
    ax.annotate(str(count), xy=(qualities.index(quality), quality))

plt.title('Video 1, Heuristic, Quality over time')
plt.show()


