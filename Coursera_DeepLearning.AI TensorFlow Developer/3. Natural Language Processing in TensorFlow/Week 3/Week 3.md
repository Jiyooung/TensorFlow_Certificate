# 3. Natural Language Processing in TensorFlow

## WEEK 3 : Sequence models

### 학습 내용
TensorFlow에서 시퀀스 모델을 구현하는 방법<br>
RNN이나 Recurrent Nearative Network와 같은 모델<br>
<br>

신경 네트워크를 위해서, 단어의 순서를 고려하기 위해서, 사람들은 이제 특수 신경 네트워크 아키텍처를 사용<br>
- RNN과 같은 것들 GIO 또는 LSTM
  - RNN
    context가 timestamp에서 timestamp까지 유지된다는 것은 정말 흥미로움<br>

  - LSTM
    Cell State를 가지고 있고, Cell State는 아주 긴 timestamp을 위해 컨베이어 벨트와 비슷<br>

[Learn more Deep RNNs in coursera](https://www.coursera.org/lecture/nlp-sequence-models/deep-rnns-ehs0S)<br>

[Learn more LSTMs in coursera](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)<br>


### LSTM
LSTM = RNN Update <br>
LSTM에는 Cell State 존재<br>
Cell State : 단방향 or 양방향<br>
    Cell State는 아주 긴 timestamp을 위해 컨베이어 벨트와 비슷<br>

[IMDB Subwords 8K with Single Layer LSTM](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201a.ipynb)<br>

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```    
- tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
Bidirectional : cell state를 양방향으로 만듬<br>
LSTM를 64로 설정했지만 양방향의 출력으로 128개가 됨<br>

[IMDB Subwords 8K with Multi Layer LSTM](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201b.ipynb)<br>
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

#### 단일 LSTM VS 멀티 LSTM
acc 훈련 곡선이 매끄러워짐<br>
단일 LSTM보다 2계층 LSTM의 곡선이 더 부드럽고 나음<br>
<br>

LSTM을 하고 안하고의 차이<br>
하면 정확도와 손실이 향상됨<br>
검증의 정확도와 손실은 안좋아짐<br>
\> LSTM overfitting<br>

### 컨볼루션 사용

[IMDB Subwords 8K with 1D Convolutional Layer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb)<br>

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- tf.keras.layers.Conv1D(128, 5, activation='relu')
컨볼루션 추가<br>
5 단어마다 128 필터를 가지고 있음<br>
max_length = 120 이므로 120에서 5단어 길이 필터는 앞뒤로 2단어를 잘라네 116으로 남겨둠<br>

컨볼루션 추가로 정확도가 향상됨<br>
검증 손실도 상승<br>
### RNN
context가 timestamp에서 timestamp까지 유지된다<br>

[IMDB Reviews with GRU (and optional LSTM and Conv1D)](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=nHGYuU4jPYaj)<br>

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

텍스트를 사용하면 이미지 작업보다 overffing 발생 가능성 높음<br>

### 전반적인 비교
- Flatten 사용
    171,533개의 param 존재, epoch 당 5초
- LSTM 사용
    30,129개의 params 존재, epoch 당 43초<br>
    LSTM 사용하면 훈련 시간은 더 걸리고, 정확도는 더 좋지만 overfitting 존재
- RNN 사용(양방향)
    169,997 개의 params 존재, epoch 당 20초<br>
    정확도 좋음, 검증 나쁘지 않음, overfitting 존재
- 컨볼루션 Conv1D 사용
    171,149 개 param 존재, epoch 당 6초<br>
    정확도 100, 검증 83, overfitting 존재

### 다양한 유형의 시퀀스 모델을 탐색해보기
[Sarcasm with Bidirectional LSTM](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202.ipynb#scrollTo=g9DC6dmLF8DC)<br>
<br>

[Sarcasm with 1D Convolutional Layer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202c.ipynb#scrollTo=g9DC6dmLF8DC)<br>
<br>
<br>

[Exercise 11 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Question.ipynb)<br>
<br>

[Exercise 11_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb)<br>
<br>

[Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140)<br>
<br>

[Embeddings Learn More](https://nlp.stanford.edu/projects/glove/)<br>