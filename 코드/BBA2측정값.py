import matplotlib.pyplot as plt
import pandas as pd


# 데이터 불러오기
df = pd.read_table('bba2.txt', sep=' ', header=None)

# x, y1, y2 데이터 추출하기
time = (df[0]*100)/1000
queuedByte = df[1]
queuedTime = df[2]
bandWith = df[3]
currentRate = df[4]
currentLevel = df[5]
getMaxLevel = df[6]
mediaEngineStatus = df[7]
pausedTime = df[8]
downLoadBytes = df[9]
cpuPercent = df[10]
memoryPercent = df[11]
memoryRss = df[12]
memoryVms = df[13]
segmentRequest = df[14]
stopSegmentRequest = df[15]


# 그래프 그리기
fig, ax1 = plt.subplots()

# 첫 번째 y축
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('currentRate', color=color)
ax1.plot(time, currentRate, color=color, label='currentRate')
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 y축
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('bandWith', color=color)
ax2.plot(time, bandWith, color=color, label='bandWith')
ax2.tick_params(axis='y', labelcolor=color)

# 세 번째 y축
ax3 = ax1.twinx()
color = 'tab:green'
ax3.spines["right"].set_position(("axes", 1.2))
ax3.plot(time,queuedTime, color=color, label='queuedTime')
ax3.tick_params(axis='y', labelcolor=color)

# 범례 추가
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower center')
# 그래프 출력
plt.show()
