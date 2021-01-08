# 2. Convolutional Neural Networks in TensorFlow

## WEEK 2 : Convolutional Neural Networks in TensorFlow

## Image Augmentation
작은 데이터 집합을 보다 효과적으로 만들기 위해 사용할 수 있는 도구<br>
overfitting 방지<br>

[Learn more Keras](https://github.com/keras-team/keras-preprocessing)<br>

[Image Augmentation implementation in Keras](https://keras.io/api/preprocessing/image/)<br>

[Week 2 실습_Dog_Cat_Augmentation](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb)<br>

### ImageDataGenerator
```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```
- rescale=1./255

- rotation_range=40
이미지를 임의로 회전할 0-180도 범위<br>
0도에서 40도 사이의 임의의 양만큼 회전<br>

- width_shift_range=0.2, height_shift_range=0.2
프레임 안쪽으로 이미지를 이동<br>
많은 사진은 피사체가 가운데에 위치<br>
이미지 크기의 비율로 피사체를 얼마나 무작위로 이동해야하는지 지정<br>
수직 또는 수평으로 20% 정도 상쇄<br>

- shear_range=0.2
이미지의 지정된 부분까지 임의의 양만큼 이미지를 전단<br>
x 축을 따라 기울이기<br>
20% 까지 전단<br>

- zoom_range=0.2
이미지를 확대<br>
확대/축소는 이미지 크기의 최대 20% 까지 임의의 양<br>

- horizontal_flip=True
수평 뒤집기, 이미지가 무작위로 반전<br>

- fill_mode='nearest'
작업에 의해 손실되었을 수 있는 픽셀을 채움<br>
픽셀의 이웃을 사용하여 균일성을 유지하려고 시도<br>

### Zoom Augmentation 적용
[ImageDataGenerator_zoom](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb)<br>
이미지에 확대만 추가해도 훈련 & 테스트 정확도가 전보다 오름<br>

Augmentation이 훈련 이미지에 무작위 요소를 도입한다는 것 > 정확도 1에 수렴<br>
유효성 검사 집합이 동일한 무작위성을 가지지 않으면(너무 무작위로 적용되면) 정확도의 변동성이 큼<br>

[pixabay](https://pixabay.com/photos/bed-dog-animals-pets-relax-1284238/)

[Exercise 6 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%206%20-%20Cats%20v%20Dogs%20with%20Augmentation/Exercise%206%20-%20Question.ipynb)