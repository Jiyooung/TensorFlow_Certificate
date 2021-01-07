 # 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## WEEK 4

### train_generator
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        # This is the source directory for training images
    '/tmp/horse-or-human/', 
        # All images will be resized to 150x150
    target_size=(300, 300),  
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
```
- 신경망을 훈련하려면 입력 데이터가 모두 동일한 크기 여야하므로 이미지의 크기를 조정하여 일관성을 유지

- target_size=(300, 300)
이미지가 로드 될 때 크기가 조정된다는 것 > 전처리 필요 X

### Training
```
history = model.fit(
    train_generator,
    steps_per_epoch=8,  
    epochs=15,
    verbose=1,
    callbacks=[callbacks])
```

### validation_generator
```
# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    train_dir,         # This is the source directory for training images
    target_size=(300, 300), # All images will be resized to 300x300
    batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
```
유효성 검사 생성기(validation_generator)는 테스트 이미지가 포함 된 하위 디렉토리를 포함하는 다른 디렉토리를 가리키는 것을 제외하고는 정확히 동일해야합

#### Conv2D 코드
```
input_shape=(300, 300, 3)
```
Every Image will be 300x300 pixels, with 3 bytes to define color<br>
픽셀 당 3 바이트<br>
빨간색 바이트, 초록색, 파란색 채널용 바이트 1바이트, 일반적인 24비트 색상 패턴<br>

클래스 당 하나의 뉴런이 있었지만 이제는 두 클래스에 대해 하나의 뉴런만 있다는 것

```
Dense(1, activation='sigmoid')
```
1 : 이진 분류의 경우 한 클래스에 대해 0에 가까운 값과 다른 클래스에 대해 1에 가까운 값을 가진 하나의 항목만 포함<br>

sigmoid : 시그모이드(sigmoid)가 바이너리 분류에 적합한 다른 활성화 함수를 사용<br>
> 이전과 동일한 softmax 함수를 사용할 수 있지만 바이너리의 경우 좀 더 효율적<br>

[Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
[Binary Classification](https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s)


### 인간과 말을 분류하는 회선 신경망을 구축
[Week 4 실습_horse_human](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=RXZT2UsyIVe_)


### 유효성 검사 집합을 추가하고 유효성 검사 집합의 정확도를 자동으로 측정하도록 하는 방법
[Week 4 실습_유효성 검사](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)

### 빠른 교육을 위해 데이터 압축이 미치는 영향
300x300 \> input_shape=(150, 150, 3)<br>

validation_generator \>\> target_size = (150,150)<br>

- You’re overfitting on your training data
이미지 크기 줄여서 하면 빠르지만 테스트 정확도는 떨어짐<br>

[Week 4 실습_데이터 압축](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb)<br>


[Exercise 4 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%204%20-%20Handling%20Complex%20Images/Exercise%204-Question.ipynb)<br>

매우 작은 데이터 세트가있을 때 과잉 피팅 (over-fitting) 이라는 오류가 발생 가능<br>