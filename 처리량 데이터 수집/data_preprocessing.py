import os

### 파일 전처리 문자제거, 단위변환, 이동평균 데이터 분리

def get_filelist(path):
    file_list = os.listdir(path)
    file_list.sort()
    return file_list

count = 0

with open("data/dataset6(v1).txt", 'rt', encoding='utf-8-sig') as data:
    lines = data.readlines()

# count = len(lines)


f = open("data/video1thp.txt", 'w')


for i in lines:
    token = i.split(',')
    avg = float(token[0].replace('null', '0'))
    avg /= 1000
    f.write(str(count))
    f.write(' ')
    data = token[-1].replace(']','')
    data = data.replace('[', '')
    data = data.replace('undefined', '0')
    data = float(data)
    data /= 1000

    f.write(str(data) + ' ' + str(avg) + '\n')

    count += 1

f.close()


# 처리량 평균 값 1칸 밀기

# 파일 열기
with open('data/video1thp.txt', 'r') as f:
    lines = f.readlines()

# 내용 변경
new_lines = ['0 ' + lines[0].split()[1] + ' 0\n']
for i in range(1, len(lines)):
    new_lines.append(str(i) + ' ' + lines[i].split()[1] + ' ' + lines[i-1].split()[2] + '\n')

# 변경된 내용 저장
with open('data/video1_data_t23.txt', 'w') as f:
    f.writelines(new_lines)

