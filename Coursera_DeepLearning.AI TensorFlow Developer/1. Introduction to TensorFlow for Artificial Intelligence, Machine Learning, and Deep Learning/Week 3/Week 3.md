# 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## WEEK 3

### Convolution
A technique to isolate features in images : 이미지의 특징을 분리하는 기술<br>
필터가 3 x 3 인 경우 이미지에서 3 x 3개의 그리드를 가져와서 수행<br>
동일한 위치에 있는 필터와 픽셀 값을 곱하기 \> 곱한 값들을 모두 더하기 = 픽셀의 새 값

- 수직선을 뽑아내는 필터(3x3)
```
    -1  0   1
    -2  0   2
    -1  0   1
```
- 수평선을 뽑아내는 필터(3x3)
```
    -1  -2  -1
     0   0   0
     1   2   1
```
[Convolution Code](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)<br>
[Learn more](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)<br>

### Pooling layers
A technique to reduce the information in an image while maintaining features : 특징을 유지하면서 이미지의 정보를 줄이는 기술<br>
단순한 풀링은 이미지를 압축하는 방법<br>
4x4 이미지 \> 한 번에 4 픽셀의 이미지, 즉 현재 픽셀과 그 아래에 있는 이웃 픽셀 선택 \> 네 가지 중에서 가장 큰 값을 선택<br>
컨볼루션에 의해 강조 표시된 기능이 보존되고 이미지의 크기가 동시에 쿼터링 됨

[Pooling Code](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)<br>

### 코드 적용
심층 신경망(Deep Neural Network)을 Convolution 신경망으로 전환하는 방법<br>
Convolution layer 추가 \> 원시 픽셀 대신 Convolution 결과에 대해 네트워크 훈련
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
                        input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```
> tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))<br>
keras에게 64 개의 필터를 생성하도록 요청<br>
필터는 3 x 3<br>
활성화는 relu<br>
입력 이미지 28x28<br>
1은 우리가 색 심도에 대해 단일 바이트를 사용하여 집계한다는 것을 의미<br>

> tf.keras.layers.MaxPooling2D(2, 2)<br>
최대 값을 취할 것이기 때문에 MaxPooling2D 사용<br>
2x2 > 4 픽셀마다 선택<br>

#### model.summary()
모델의 레이어를 검사하고 회선을 통해 이미지의 이동을 볼 수 있음<br>
중요한 것은 Output Shape!<br>

- 28x28이미지인데 왜 26,26이 되었는가?<br>
핵심은 필터가 3 x 3 필터라는 것<br>
테두리 픽셀들은 주위에 1 픽셀 여백을 사용할 수 없으므로 x에서 2 픽셀이 작아지고 y에서는 2 픽셀이 작아짐<br>
5x5 필터면 x - 4, y - 4<br>

> tf.keras.layers.Flatten()<br>
5 x 5 새로운 64 개의 이미지<br>
Flatten \> 1600 = 새로운 병합 레이어<br>
컨볼루션 2D 레이어를 정의할 때 설정한 매개변수의 영향을 받음<br>

[Week 3 실습](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)


### 시각화
```
import cv2
from scipy import misc
i = misc.ascent()
```
SciPy에서 기타 라이브러리를 가져옴<br>
misc.ascent는 우리가 가지고 놀 수 있는 멋진 이미지를 반환<br>

[Week 3 실습_시각화](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)<br>

[More information](https://lodev.org/cgtutor/filtering.html)<br>

[Exercise 3 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Question.ipynb)



