# 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## WEEK 2

### Fashion MNIST
- 70k Images
- 10 Categories
- Images are 28x28
- Can train a neural net!
- [Learn more](https://github.com/zalandoresearch/fashion-mnist)

Machine Learning depends on having good data to train a system with<br>

```
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

training data = 60000개<br>
test data = 10000개<br>

#### training_images에 training_labels 붙이는 이유
1. 컴퓨터가 텍스트보다 숫자로 더 잘 작동한다
2. bias를 줄이는 데 도움이 된다.

### 데이터 정규화
우리의 이미지는 0에서 255 사이의 값을 가지고 있지만 신경망은 정규화 된 데이터로 더 잘 작동함 <br>\> 단순히 255로 모든 값을 나누어 0과 1 사이 값으로 변경하기
```
training_images = training_images / 255.0
test_images = test_images / 255.0
```

### 모델 디자인
데이터 모양의 입력 계층과 클래스 모양의 출력 계층이 있고 둘 사이의 역할을 파악하려는 숨겨진 레이어가 하나 있음
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),         # Flatten : 28x28 사각형 > simple linear array
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)    # 10개의 뉴런 > 데이터 세트에 10가지 종류의 의류 존재
])
```
### 신경망이 수행하는 일
28*28 > label<br>
y = w0 * x0 + w1 * x1 + w2 * x2 + ... + wN * xN = label<br>

학습하기
```
model.compile(optimizer = tf.train.AdamOptimizer(),
                loss = 'sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
```

[Week 2 실습](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=WzlqsEzX9s5P)

### Callback()
1 epoch이 끝날 때마다 산출되는 'loss' or 'accuracy'값이 정해진 값 이상/이하로 내려가면 training 중단하기
```
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()    # 별도의 클래스로 구현
.
.
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```

[Callback python code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb#scrollTo=N9-BCmi15L93)
