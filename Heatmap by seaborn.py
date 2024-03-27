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

# seaborn을 이용한 히트맵 그리기
ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap='Greys', linewidths=0.5)

# 글씨 가로로 돌리기
plt.xticks(rotation = 45)
plt.yticks(rotation = 0)
plt.xlabel('Estimated Label')
plt.ylabel('True Label')
plt.title('confusion matrix of face expression recognition', fontsize=20)
plt.show()