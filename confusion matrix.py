import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix 데이터
conf_matrix = np.array([[976, 0, 0, 0, 6, 18, 0],
                         [0, 997, 0, 0, 3, 0, 0],
                         [1, 0, 982, 0, 0, 6, 11],
                         [1, 2, 2, 995, 0, 0, 0],
                         [14, 0, 0, 0, 975, 11, 0],
                         [17, 0, 0, 0, 5, 978, 0],
                         [0, 0, 3, 0, 0, 0, 997]])

# 클래스 레이블
classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# seaborn heatmap을 이용하여 시각화
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # 폰트 크기 설정
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Estimated Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
