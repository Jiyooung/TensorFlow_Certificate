# 3. Natural Language Processing in TensorFlow

## WEEK 1 : Sentiment in text

### 학습 내용
텍스트를 처리하는 방법<br>
텍스트를 로드하는 방법을 배우고 사전 처리 및 데이터 설정해서 신경망에 공급하는 것을 배움<br>
ASCII 값 이용 <br>
TensorFlow와 Care Ask는 이 작업을 매우 간단하게 하는 API를 제공<br>

### tokenizer
자연어 처리, 토큰 관리, 텍스트 전환 등의 모든 작업을 수행<br>

문장들을 토큰화 <br>
[Week 1 실습_tokenizer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%201.ipynb)<br>

```
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```
- tokenizer = Tokenizer(num_words = 100)
하이퍼 파라미터를 설정하면 tokenizer가하는 일은 볼륨별로 상위 100 단어를 가져 와서 인코딩하는 것<br>
많은 양의 데이터를 처리 할 때 편리한 방법<br>
  - num_words = 100
  가장 일반적인 단어 100개 또는 여러분이 실제로 여기에 넣은 어떤 값이든지 필요<br>
  실제로 아무런 영향을 미치지 않음<br>

단어가 적을수록 영향이 최소화되고 훈련 정확도는 높아지지만 훈련 시간은 길지만 주의해서 사용<br>

- tokenizer.fit_on_texts(sentences)
단어 index을 생성하고 토크 나이저를 초기화<br>
데이터를 가져 와서 인코딩<br>

- word_index = tokenizer.word_index
tokenizer는 키 값이 포함 된 사전을 반환하는 단어 index 속성을 제공<br>
키는 단어, 값은 해당 단어에 대한 토큰<br>
공백과 쉼표와 같은 구두점이 제거됨<br>
대소문자를 구분하지 않음<br>

```
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
```
- sequences = tokenizer.texts_to_sequences(sentences)
토큰기를 호출하여 시퀀스에 대한 텍스트를 가져오고, 이를 제게 일련의 시퀀스로 변환<br>
정수 목록으로 인코딩 된 문장 목록이 있으며 토큰은 단어를 대체<br>

시퀀스로 호출된 텍스트는 어떤 문장이든 쓸 수 있다는 것<br>

### Padding

[Week 1 실습_pad_sequences](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%202.ipynb#scrollTo=rX8mhOLljYeM)<br>

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 100)

# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)

padded = pad_sequences(test_seq, maxlen=10)
```
- 'my dog loves my manatee'
단어 지수에 없어서 [1, 3, 1] = my dog my 로 해석됨<br>

```
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
```
- oov_token="<OOV>"
속성 oov 토큰을 추가<br>
사전에 없는 단어는 oov로 표시<br>

- 'my dog loves my manatee'
단어 지수에 없어서 [2, 4, 1, 2, 1] = my dog oov my oov 로 해석됨

- padded = pad_sequences(test_seq, maxlen=10)
이미지도 사이즈 지정한 것처럼 텍스트도 크기 지정 필요<br>
크기의 균일 성이 일정해야하므로 패딩 사용<br>
채우기 또는 자르기를 사용하여 모든 문장을 동일한 길이로 만들 수 있다<br>

  - padding = 'post'
    지정 길이보다 문장이 작으면 뒤에 0으로 채움<br>
    지정 안하면 앞에 0으로 채움<br>

  - maxlen=10
    최대 단어 길이 지정, 지정 안하면 데이터 중 가장 긴 문장 길이로 지정<br>
    문장이 최대 길이보다 길면 문장 앞부분의 정보 잃음<br>

  - truncating='post'
    문장이 최대 길이보다 길 때 문장 뒷부분의 정보 잃도록 지정<br>


### sarcasm detection 데이터 사용
[dataset for sarcasm detection](https://rishabhmisra.github.io/publications/)

1. 비꼬는 것 (is_sarcastic)
   '1'이란 비아냥거림, '0'이란 농담<br>
2. 제목 (headline)
3. 제목이 설명하는 기사에 대한 링크 (article_link)


[Week 1 실습_sarcasm](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb)<br>

[json file url](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json)<br>

```
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```
- datastore = json.load(f)
헤드 라인, URL 및 is_sarcastic 레이블의 세 가지 데이터 유형 목록이 포함 된 목록 get<br>

```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
print(word_index) # 29,657 개의 단어 존재

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)
```

[BBC Datasets](http://mlg.ucd.ie/datasets/bbc.html)<br>
<br>

[stopwords data](https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js)<br>
<br>

[Exercise 9 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-question.ipynb)<br>

[Exercise 9_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-answer.ipynb)<br>



