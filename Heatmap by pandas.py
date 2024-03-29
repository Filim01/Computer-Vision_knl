import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# 주어진 혼동 행렬
conf_matrix = np.array([[976, 0, 0, 0, 6, 18, 0],
                         [0, 997, 0, 0, 3, 0, 0],
                         [1, 0, 982, 0, 0, 6, 11],
                         [1, 2, 2, 995, 0, 0, 0],
                         [14, 0, 0, 0, 975, 11, 0],
                         [17, 0, 0, 0, 5, 978, 0],
                         [0, 0, 3, 0, 0, 0, 997]])

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# 혼동 행렬을 백분율로 변환
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 데이터프레임 생성 -> pandas는 DateFrame 형식으로 데이터를 다루기 때문에
df_percent = pd.DataFrame(conf_matrix_percent, index=labels, columns=labels)
# 히트맵 그리기
plt.imshow(df_percent, cmap='Blues', interpolation='nearest')

# 레이블 설정
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)

# 텍스트 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            plt.text(j, i, df_percent.iloc[i, j], ha='center', va='center', color='white')
        else:
            plt.text(j, i, df_percent.iloc[i, j], ha='center', va='center', color='black')
        # plt.text(j, i, df_percent.iloc[i, j], ha='center', va='center', color='Black')
        # 교수님께 드리는 질문 왜 pandas에서는 Greys로 하면 오류가 생기나요? ha(수평정렬) , va(수직정렬) 때문인지 질문하고 싶습니다!

#아래는 동일하게 그래프를 설정해주기 위해서
plt.xticks(rotation = 45)
plt.yticks(rotation = 0)
plt.xlabel('Estimated Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Face Expression Recognition(pandas)', fontsize=20)
plt.show()