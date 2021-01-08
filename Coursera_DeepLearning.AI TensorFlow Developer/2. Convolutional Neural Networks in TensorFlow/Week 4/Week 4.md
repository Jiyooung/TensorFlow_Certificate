# 2. Convolutional Neural Networks in TensorFlow

## WEEK 4 : Multiclass Classifications

### CGI
컴퓨터 그래픽을 사용하여 이미지를 생성<br>
CGI를 이용하여 데이터가 부족한 딥 러닝 알고리즘에 공급하는 것을 고려하고 있다<br>

[CGI techniques로 만은 Rock Paper Scissors Dataset](http://www.laurencemoroney.com/rock-paper-scissors-dataset/)<br>
white background<br>
Each image is 300×300 pixels in 24-bit color<br>
training_set, test_set, validation_set \> here 누르면 다운로드 가능<br>

### 다중 클래스 데이터 적용하기
training_set : 각 디렉토리에 840개씩<br>
test_set : 각 디렉토리에 124개씩<br>
validation_set : 섞어서 33개<br>

[Week 4 실습_가위_바위_보](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb)


```
train_generator = training_datagen.flow_from_directory(
	...,
	class_mode='categorical'
)
```

```
model = tf.keras.models.Sequential([
    ...,
    tf.keras.layers.Dense(3, activation='softmax')
])
```

```
model.compile(loss = 'categorical_crossentropy',..)
```

### overfitting 피하기
1. Augmentation
2. Dropout
3. Transfer Learning

[Exercise 8 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%208%20-%20Multiclass%20with%20Signs/Exercise%208%20-%20Question.ipynb#scrollTo=wYtuKeK0dImp)

[sign-language-mnist](https://www.kaggle.com/datamunge/sign-language-mnist)



