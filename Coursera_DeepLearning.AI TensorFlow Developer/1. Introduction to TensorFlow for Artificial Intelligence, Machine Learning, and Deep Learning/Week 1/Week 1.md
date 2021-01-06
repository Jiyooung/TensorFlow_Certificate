# 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## WEEK 1

**“machine learning”** is all about a computer learning the patterns that distinguish things.<br>

파이썬과 텐서 플로우와 keras라는 텐서 플로우의 API를 사용하여 작성됩니다.
Keras를 사용하면 신경망을 쉽게 정의 할 수 있습니다. 신경망은 기본적으로 패턴을 배울 수있는 일련의 기능입니다.<br>

Google Colaboratory 에서 코드 작성해 보기<br>
[FAQ](https://research.google.com/colaboratory/faq.html)<br>
[코드](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=DzbtdRcZDO9B)


Dense : 연결된 뉴런의 레이어 정의, Dense 한개면 하나의 층, 하나의 단위, 하나의 뉴런<br>
optimizer : 이를 측정 한 다음 다음 추측을 파악하는 최적화 프로그램에 데이터를 제공<br>
(Generates a new and improved guess : 새롭고 향상된 추측을 생성합니다.)<br>
sgd: 확률 그라데이션 하강을 나타냄<br>
loss : 손실 함수를 사용하여 추측이 얼마나 좋은지 또는 얼마나 나쁘게 수행되었는지에 대해 측정<br>
(Measures how good the current ‘guess’ is : 현재의 '추측'이 얼마나 좋은지 측정)<br>
mean_squared_error : 평균 제곱 오차<br>
> 다른 함수들은 Tensorflow 설명서 참고
```
model.fit(xs, ys, epochs=500)		# 500 epochs = 500번 반복, 학습
model.predict([10.0])			    # x가 10일 경우 예측
```
Y = 2X-1<br>

X가 10일 때 머신러닝을 통해 Y가 답인 19에 매우 가깝지만 정확히 19가 아닌 값을 반환한다는 것을 알 수 있습니다. <br>
왜? 궁극적으로 두 가지 주된 이유가 있습니다.<br>
첫 번째는 아주 적은 데이터를 사용하여 훈련했다는 것입니다. 6개의 데이터 셋만 있습니다. 이 6개는 선형이지만 모든 X에 대해 관계가 Y가 2X에서 1을 뺀 값과 같을 것이라는 보장은 없습니다. <br>
X가 10이면 Y가 19와 같을 확률이 매우 높지만 신경망은 양수가 아닙니다. 그래서 Y에 대한 현실적인 가치를 알아낼 것입니다. 이것이 두 번째 주된 이유입니다. 신경망을 사용할 때 모든 것에 대한 답을 알아 내려고 할 때 확률을 처리합니다.<br>


[Week1 실습](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)




