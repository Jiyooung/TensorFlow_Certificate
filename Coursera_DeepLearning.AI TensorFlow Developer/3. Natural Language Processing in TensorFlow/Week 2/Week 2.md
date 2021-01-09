# 3. Natural Language Processing in TensorFlow

## WEEK 2 : Word Embeddings
 
### 학습 내용
1. Embeddings 사용하는 방법<br>
2. 해당 시각화를 제공하는 분류자를 만드는 방법<br>

Embeddings라는 것을 사용하여 이 숫자를 가져 와서 그로부터 감정을 확립하기 시작하여 텍스트를 분류하고 나중에 예측<br>
문자 분류를 위한 신경망을 학습<br>
사전 훈련 된 단어 다운로드 가능<br>



### Embeddings
단어와 관련 단어가 다차원 공간에서 벡터로 묶여있다는 개념<br>
목적 : It is the number of dimensions for the vector representing the word encoding<br>
<br>

두 가지 범주<br>
 - positive 1
 - negative 0
 
### IMDB 분류
[IMDB reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
50,000 movie reviews which are classified as positive of negative. 

#### TensorFlow Data Services 또는 TFTS라는 라이브러리
많은 데이터 세트와 다양한 범주가 포함
다양한 유형, 특히 이미지 기반에 대한 다양한 데이터 세트가 있음

[Week 2 실습_tfds API_IMDB](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb)

```
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
```
#### 데이터 로드
```
import numpy as np

# 25000개, 25000개
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```

```
tf.Tensor(1, shape=(), dtype=int64)
```
1은 긍정적인 리뷰, 0은 부정적인 리뷰를 나타냄<br>

#### 토큰화
```
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
```

```
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])
```

#### 신경망 정의
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```
- tf.keras.layers.Embedding(vocab_size, embedding_dim
TensorFlow에서 텍스트 감정 분석의 핵심<br>
  - vocab_size
    10000

  - embedding_dim
    16

  - input_length=max_length
    120

- tf.keras.layers.Flatten() => <br>
    \> tf.keras.layers.GlobalAveragePooling1D() => 16<br>
벡터 전체의 평균을 평평하게 만듬<br>
Flatten보다 더 간단하고 조금 더 빠르지만 조금 덜 정확<br>

#### Training
```
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
```
acc이 1이고 val_acc은 0.80대<br>
val_acc는 아주 좋지만 acc은 overfitting이 발생한 것<br>


#### Embeddings
```
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
```
10,000 개의 단어, 16차원 배열<br>

#### 시각화
벡터를 작성하고 메타 데이터 자동 파일. TensorFlow Projector는이 파일 형식을 읽고 이를 사용하여 3D 공간에 벡터를 플롯하여 시각화<br>
```
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```
vector.TSV, meta.TSV 파일이 다운로드 됨<br>

- out_m.write(word + "\n")
메타 데이터 배열에 단지 단어를 적음<br>

- out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
벡터 파일에 우리는 단순히 삽입물의 배열에 있는 각 항목의 값, 즉 이 단어에 대한 벡터의 각 차원의 계수를 작성<br>

#### Colab에서 실행 시 수행할 코드
```
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
```

#### 결과 렌더링
[결과 확인 사이트](http://projector.tensorflow.org/) \> TensorFlow Embedding Projector \> `Load` 누르기 > 첫 번째에는 vector.TSV, 두 번째는 meta.TSV를 로드 \> `Sphereize data` 체크 \> 이진 데이터의 클러스터링 볼 수 있음<br>


### sarcasm dataset

[Week 2 실습_sarcasm](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb)<br>
20000개 학습, 나머지 검증<br>

#### 하이퍼 파라미터
```
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
```


#### 데이터 로드
```
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```


#### 토큰화
```
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
```


#### TensorFlow 2.x 이면 수행해야 할 코드
```
# Need this block to get it to work with TensorFlow 2.x
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```


#### 신경망 정의
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```


#### Training
```
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
```


#### 시각화
```
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```
training의 accuracy와 loss는 좋음<br>
testing의 accuracy와 그럭저럭 괜찮지만 loss는 증가<br>

- 어떻게 조정?
hyper parameter를 조정하기<br>
vacab_size = 1000 (감소)<br>
max_length = 16   (감소)<br>
\> accuracy는 높지 않지만 증가했고 loss은 평평해져서 이전보다 좋아 보임<br>
<br>

+ embedding_dim = 32 (증가)<br>
\> 별 차이 없음<br>
<br>

\>\> 이런 식으로 loss가 급격히 증가하지 않고 90 % 이상의 훈련 정확도를 제공하는 조합을 찾기<br>

[TensorFlow Datasets Github](https://github.com/tensorflow/datasets/tree/master/docs/catalog)<br>
<br>

[imdb_reviews](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md)<br>
<br>

[TensorFlow Datasets Catalog](https://www.tensorflow.org/datasets/catalog/overview)<br>
<br>


단어의 순서가 그 존재만큼이나 중요 할 수 있음을 보여주기 위해 Subwords를 사용<br>

[Week 2_Subwords](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb)<br>

#### 데이터 로드
```
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
```

#### 문자열을 인코딩하거나 디코딩하는 방법
```
sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
```

#### 토큰 자세히 보기
```
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))
```
토큰 자체를 보고 싶다면 각 요소를 가져 와서 decode, 토큰에 값을 표시<br>
대소 문자를 구분하고 문장 부호를 표시<br>

#### 신경망 정의 
```
embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
IMDB를 분류하는 방법<br>
Flatten() 사용하면 TensorFlow 충돌 발생!!<br>


[Exercise 10 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Question.ipynb)<br>

[Exercise 10_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb)<br>