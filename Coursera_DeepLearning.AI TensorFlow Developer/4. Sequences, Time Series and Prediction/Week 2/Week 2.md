# 4. Sequences, Time Series and Prediction

## WEEK 2 : Deep Neural Networks for Time Series

## 학습 내용
시계열을 학습 및 검증 데이터로 나누는 방법을 이해하는 중요한 작업을 포함하여 시계열 분류를 위해 DNN을 사용하는 방법<br>

[Week 2 실습_Preparing features and labels](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%201.ipynb)<br>

```
dataset = dataset.window(5, shift=1, drop_remainder=True)
```
- drop_remainder=True
개수가 부족한 부분은 drop<br>

#### features와 labels(마지막 것)로 분할
```
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
```

#### 데이터 셔플 
```
dataset = dataset.shuffle(buffer_size=10)
```
buffer_size = 10 = 우리가 가진 데이터<br>

# 데이터를 두 개의 세트로 배치 
```
dataset = dataset.batch(2).prefetch(1)
```
한 번에 두 개의 x와 두 개의 y로 일괄 처리<br>


[Week 2 실습_Single layer neural network](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%202.ipynb)<br>

```
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```
- dataset = tf.data.Dataset.from_tensor_slices(series)
데이터 집합을 사용하여 시리즈에서 데이터 집합을 만드는 것<br>

- dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
데이터 병합 \> 평평해짐 \> 셔플하기 쉬움<br>

#### 간단한 선형 회귀를 수행하는 코드
```
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

print("Layer weights {}".format(l0.get_weights()))
```
- l0 = tf.keras.layers.Dense(1, input_shape=\[window_size])
input_shape이 window_size인 단일 Dense 레이어<br>

- model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
손실을 MSE로 설정하여 평균 오류 손실 제곱 함수(mse)를 사용<br>
최적화 프로그램은 확률 그라데이션 하강(SGD)을 사용<br>
LR 및 모멘텀과 같은 매개 변수를 설정하여 초기화<br>

- print("Layer weights {}".format(l0.get_weights()))
학습된 가중치를 인쇄<br>
첫 번째 배열의 각 값은 x의 20 값에 대한 가중치<br>
두 번째 배열의 값은 b 값<br>

[Week 2 실습_Deep neural network](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%203.ipynb)<br>

#### 3개의 layers
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])
```

```
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)
```
- lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
학습 속도를 조정하는 콜백<br>
학습 속도를 eposh 수에 따라 변경<br>

```
optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=500, verbose=0)
```
- optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
learning rate 수정 > 조금 더 훈련

#### Plot all but the first 10
```
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
print(plot_loss)
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
plt.show()
```

[Exercise 14 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Question.ipynb)<br>

[Exercise 14_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Answer.ipynb)<br>
