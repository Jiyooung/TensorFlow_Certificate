# 3. Natural Language Processing in TensorFlow

## WEEK 4 : Sequence models and literature

## 학습 내용
1. 단어에 대한 예측을 얻는 방법
2. 예측을 기반으로 새 텍스트를 생성하는 방법

[Week 4 실습_하나의 문자열로 실습](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%201%20-%20Notebook.ipynb)<br>
<br>

모든 가사가 담긴 이 하나의 긴 문자열<br>
고유 한 단어의 총 수 = 263개<br>
```
corpus = data.lower().split("\n")
```
개행문자로 나누고 소문자로 변환<br>

#### Training
```
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]

	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```
- token_list = tokenizer.texts_to_sequences([line])[0]
토큰 목록 생성<br>

- for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)
단순하게 처음 두 단어, 세 단어,... , 문장 전체로 나눔<br>
8단어 문장이면 input_sequences 가 7개의 리스트로 나옴<br>

- max_sequence_len = max([len(x) for x in input_sequences])
가장 긴 문자의 길이 찾아 모든 시퀀스를 패딩하여 길이가 동일하도록 설정<br>

- xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
input_sequences의 7개 각각 맨 마지막은 Label(Y)이고 앞에는 Input(X)로 xs, labels 설정<br>

- ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
one-hot encode<br>
목록을 범주형으로 변환<br>

```
  model = Sequential()
  model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
# LSTM, 양방향
  model.add(Bidirectional(LSTM(20)))
# 단어 당 하나의 뉴런 존재
  model.add(Dense(total_words, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(xs, ys, epochs=500, verbose=1)
```
- model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
모든 단어 다루기, 단어의 벡터를 그리는 데 사용할 차원 수 = 100<br>
64차원<br>
입력 치수의 크기 =  가장 긴 시퀀스에서 1을 뺀 길이<br>
  - 라벨을 얻기 위해 각 시퀀스의 마지막 단어를 잘라내기 때문에 1을 뺌<br>


```
seed_text = "Laurence went to dublin"
next_words = 100
  
for _ in range(next_words):
# 다음 단어가 될 가능성이 가장 높은 단어의 토큰을 얻을 수 있음
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
# 토큰을 다시 한 단어로 바꾸고 시드 텍스트에 추가
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
```
각 단어가 예측되기 때문에 100 % 확실하지 않으면 다음 단어는 더 확실하지 않으며 다음 단어 등은 더 확실하지 않음<br>

#### 1,692 개의 문장을 가진 많은 노래가 들어있는 파일을 준비
[irish-lyrics-eof.txt](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt)<br>
<br>

[Week 4 실습_irish-lyrics-eof](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb)<br>


### 순환 신경망을 활용한 문자열 생성
[TensorFlow Tutorial_using RNN](https://www.tensorflow.org/tutorials/text/text_generation)<br>

스스로 시도해봐라!<br>
<br>

[Exercise 12 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Question.ipynb)<br>
<br>

[Exercise 12_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb)<br>
<br>
