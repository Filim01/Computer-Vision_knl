import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn은 Matplotlib을 기반으로


plt.rcParams['figure.figsize'] = [10, 8]

conf_matrix = np.array([[976, 0, 0, 0, 6, 18, 0],
                         [0, 997, 0, 0, 3, 0, 0],
                         [1, 0, 982, 0, 0, 6, 11],
                         [1, 2, 2, 995, 0, 0, 0],
                         [14, 0, 0, 0, 975, 11, 0],
                         [17, 0, 0, 0, 5, 978, 0],
                         [0, 0, 3, 0, 0, 0, 997]])

# x와 y 축의 레이블 설정 -> 숫자로 되어있기 때문에
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

#혼동 행렬을 백분율로 변환
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 백분율 값을 소수점 형식으로 포맷팅하자!
# 0 은 0.0으로 표시하고 싶으니까 != 사용해보자 그러려면 if 문을 써야겠지? -> 안되니까 conf_matrix_percent로 각 요소를 나타내는 변수 사용해서 해야할듯
conf_matrix_percent_formatted = [[('{:.3f}'.format(conf_matrix_percent[i][j]) if conf_matrix_percent[i][j] != 0 else '0.0') for j in range(len(labels))] for i in range(len(labels))]

# seaborn을 이용한 히트맵 그리기
ax = sns.heatmap(conf_matrix_percent, xticklabels=labels, yticklabels=labels, annot=conf_matrix_percent_formatted, fmt='', cmap='Blues', linewidths=0.5)

# 글씨 가로로 돌리기
plt.xticks(rotation = 45)
plt.yticks(rotation = 0)
plt.xlabel('Estimated Label')
plt.ylabel('True Label')
plt.title('confusion matrix of face expression recognition(seaborn)', fontsize=20)
plt.show()