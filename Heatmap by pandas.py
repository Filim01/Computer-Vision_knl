import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#히트맵을 그리기 위해서는 Matplotlib을 기반으로 한 imshow 함수를 사용 해야 한다.
#사용자가 직접 각 셀의 값을 텍스트로 표시해야 한다.


conf_matrix = np.array([[976, 0, 0, 0, 6, 18, 0],
                         [0, 997, 0, 0, 3, 0, 0],
                         [1, 0, 982, 0, 0, 6, 11],
                         [1, 2, 2, 995, 0, 0, 0],
                         [14, 0, 0, 0, 975, 11, 0],
                         [17, 0, 0, 0, 5, 978, 0],
                         [0, 0, 3, 0, 0, 0, 997]])


labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# 데이터프레임 생성 -> pandas는 DateFrame 형식으로 데이터를 다루기 때문에
df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
# 히트맵 그리기
plt.imshow(df, cmap='Greys', interpolation='nearest')

# 레이블 설정
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)

# 레이블 표시
for i in range(len(labels)):
    for j in range(len(labels)):
        # 교수님께 드리는 질문 왜 pandas에서는 Greys로 하면 오류가 생기나요? ha(수평정렬) , va(수직정렬) 때문인지 질문하고 싶습니다!
        if i==j :
            plt.text(j, i, str(df.iloc[i, j]), ha='center', va='center', color='white')
        else :
            plt.text(j, i, str(df.iloc[i, j]), ha='center', va='center', color='Black')

#아래는 동일하게 그래프를 설정해주기 위해서
plt.xlabel('Estimated Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Face Expression Recognition', fontsize=20)
plt.colorbar()
plt.show()