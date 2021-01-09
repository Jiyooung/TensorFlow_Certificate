# 4. Sequences, Time Series and Prediction

## WEEK 3 : Recurrent Neural Networks for Time Series

## 학습 내용
1. RNN과 LCM을 시퀀스 데이터에 적용
    RNN과 LSTM을 사용할 수 있다는 것은 훨씬 더 정확한 예측을 제공하기 위한 데이터를 반영 가능<br>
2. Lambda 레이어를 구현


### Lambda 레이어
임의의 코드를 신경망의 레이어로 효과적으로 작성<br>
명시 적 전처리 단계로 데이터를 확장 한 다음 해당 데이터를 신경망에 공급하는 대신 Lambda 계층을 사용 가능<br>
Lambda 함수는 데이터를 다시 보내는 신경망의 레이어로 구현되어 크기를 조정<br>
이제는 전처리 단계가 더 이상 별개의 단계가 아니라 신경망의 일부<br>

### RNN (Recurrent Neural Network)
시퀀스 데이터는 RNN에서 더 잘 작동하는 경향이 있음<br>
매우 유연하며 모든 종류의 시퀀스를 처리 가능<br>
RNN을 사용할 때의 전체 입력 형상이 3차원적이라는 것<br>
첫 번째 차원은 배치 크기이고 두 번째 차원은 타임 스탬프이고 세 번째 차원은 각 시간 단계에서 입력의 차원<br>
일변량 시계열 인 경우이 값은 1이 될 것이고 다변량의 경우 더 많을 것<br>
동일한 레이어를 여러 번 재사용<br>
실제로는 마지막 출력을 제외한 모든 출력을 무시<br>

<p align="center"> <img src="https://github.com/Jiyooung/TensorFlow_Certificate/blob/main/Coursera_DeepLearning.AI%20TensorFlow%20Developer/4.%20Sequences%2C%20Time%20Series%20and%20Prediction/Week%203/RNN%20Layer.PNG" alt="drawing" width="800"/>

메모리 셀이 세 개의 뉴런으로 구성되어있는 경우, 들어오는 배치 크기가 4이고 뉴런의 수는 3이기 때문에 출력 행렬은 4x3<br>
레이어의 전체 출력은 3 차원이며, 이 경우 4 x 30 x 3<br>
4는 배치 크기이고, 3은 단위 수이고, 30은 전체 단계의 수<br>

[Week 3 실습_RNN](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%202%20-%20RNN.ipynb)<br>

#### RNN 적용
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```
- tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
Lambda를 사용하여 배열을 1차원 확장<br>

- tf.keras.layers.SimpleRNN(40, return_sequences=True),
RNN 레이어의 기본 활성화 기능은 쌍곡선 탄젠트 활성화(hyperbolic tangent activation)인 tan H.<br>
tan H는 음수 1과 1 사이의 값을 출력<br>
  - return_sequences=True
    다음 레이어로 공급되는 시퀀스를 출력<br>

#### Huber() 손실 함수 도입
[More info on Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
특이치에 덜 민감한 손실 함수<br>

### LSTM
LSTM은 상태가 셀에서 셀로 전달되고 타임 스탬프로 타임 스탬프로 타임 스탬프로 전달되도록 교육의 수명 동안 상태를 유지하는 셀 상태<br>
RNN의 경우보다 전체 projection에 더 큰 영향을 미칠 수 있음<br>
상태가 단방향, 양방향 가능<br>

[Week 3 실습_LSTM](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%204%20-%20LSTM.ipynb)<br>

#### 내부 변수를 지우기
```
tf.keras.backend.clear_session()
```

#### LSTM 적용
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```
- tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
32 개의 셀이있는 양방향 LSTM 레이어를 추가<br>


[Exercise 15 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Exercise%20Question.ipynb)<br>

[Exercise 15_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Exercise%20Answer.ipynb)<br>


