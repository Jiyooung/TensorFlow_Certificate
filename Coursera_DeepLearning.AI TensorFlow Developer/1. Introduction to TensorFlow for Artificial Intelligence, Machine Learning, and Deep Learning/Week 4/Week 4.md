 # 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## WEEK 4

순서
1. model = tf.keras.models.Sequential()
2. model.compile()
3. ImageDataGenerator - train_generator, validation_generator
    train_datagen = ImageDataGenerator(rescale=1/255)<br>
    train_generator = train_datagen.flow_from_directory()<br>
4. history = model.fit_generator()

### Sequential
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
                        input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- input_shape=(300, 300, 3)
Every Image will be 300x300 pixels, with 3 bytes to define color<br>
픽셀 당 3 바이트<br>
빨간색 바이트, 초록색, 파란색 채널용 바이트 1바이트, 일반적인 24비트 색상 패턴<br>

- Dense(1, activation='sigmoid')

이전에는 클래스 당 하나의 뉴런이 있었지만 이제는 두 클래스에 대해 하나의 뉴런만 있음<br>
**1** : 이진 분류의 경우 한 클래스에 대해 0에 가까운 값과 다른 클래스에 대해 1에 가까운 값을 가진 하나의 항목만 포함<br>
**sigmoid** : 시그모이드(sigmoid)가 바이너리 분류에 적합한 다른 활성화 함수를 사용<br>
> 이전과 동일한 softmax 함수를 사용할 수 있지만 바이너리의 경우 좀 더 효율적<br>

### compile
```
model.compile(loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy'])
```
- rmsprop optimizer
lr 매개 변수를 조정하여 학습 속도를 조정할 수 있음

[Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
[Binary Classification](https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s)

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
```
train_generator = train_datagen.flow_from_directory(
    훈련이미지 폴더, 이미지 크기 지정, 배치 크기, 클래스 모드
)
```
- flow_from_directory()
The ability to easily load images for training<br>
The ability to pick the size of training images<br>
The ability to automatically label images based on their directory name<br>

- target_size=(300, 300)
이미지가 로드 될 때 크기가 조정된다는 것 > 전처리 필요 X<br>
신경망을 훈련하려면 입력 데이터가 모두 동일한 크기 여야하므로 이미지의 크기를 조정하여 일관성을 유지

- batch_size=20
2000개 이미지가 있으면 각각 20개로 구성된 100개의 배치 사용

- class_mode='binary'
2개의 클래스가 있기 때문에 binary 사용

### validation_generator
```
# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        # This is the source directory for validation images
    validation_dir,         
        # All images will be resized to 300x300
    target_size=(300, 300), 
    batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
```
유효성 검사 생성기(validation_generator)는 테스트 이미지가 포함된 폴더 지정을 제외하고는 train_generator와 정확히 동일해야합

### Training
```
history = model.fit(
    train_generator,
    steps_per_epoch=8,  
    epochs=15,
    verbose=1,
    callbacks=[callbacks])

history = model.fit(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch=8,  
    epochs=15,
    verbose=1,
    validation_steps=8)
```
model.fit or model.fit_generator 사용

### 인간과 말을 분류하는 회선 신경망을 구축
[Week 4 실습_horse_human](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=RXZT2UsyIVe_)

### 유효성 검사 집합을 추가하고 유효성 검사 집합의 정확도를 자동으로 측정하도록 하는 방법
[Week 4 실습_유효성 검사](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)

### 빠른 교육을 위해 데이터 압축이 미치는 영향
Conv2D \>\> 300x300 \> input_shape=(150, 150, 3)<br>

validation_generator \>\> target_size = (150,150)<br>

- You’re overfitting on your training data
이미지 크기 줄여서 하면 빠르지만 테스트 정확도는 떨어짐<br>

[Week 4 실습_데이터 압축](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb)<br>


[Exercise 4 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%204%20-%20Handling%20Complex%20Images/Exercise%204-Question.ipynb)<br>

매우 작은 데이터 세트가있을 때 과잉 피팅 (over-fitting) 이라는 오류가 발생 가능<br>