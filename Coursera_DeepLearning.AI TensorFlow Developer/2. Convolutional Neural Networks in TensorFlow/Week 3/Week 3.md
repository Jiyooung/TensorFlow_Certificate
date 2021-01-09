# 2. Convolutional Neural Networks in TensorFlow

## WEEK 3 : Transfer Learning

## Transfer Learning
모델을 더 빠르게 교육<br>
높은 정확도를 얻는 방법<br>

### Inception
[ImageNet](http://image-net.org/)<br>
1000 개의 다른 클래스에서 1.4 백만 개의 이미지를 가진 Imagenet의 데이터 세트에 대해 학습이 되어 있음<br>

[Week 3 실습](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)<br>

[tensorflow tutorial - 사전 학습된 ConvNet을 이용한 전이 학습](https://www.tensorflow.org/tutorials/images/transfer_learning)<br>

### Inception V3
```
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)
```
inception V3는 fully-connected layer를 가지고 있다.

- include_top = False
이를 무시하고 회선으로 바로 이동하도록 지정

```
for layer in pre_trained_model.layers:
    layer.trainable = False
```
lock or freeze a layer from retraining<br>
사전 훈련 된 모델을 인스턴스화 했으므로 레이어를 반복하고 잠글 수 있으며 이 코드로 교육 할 수 없다<br>

### 모델 요약보기
```
pre_trained_model.summary()
```

### 마지막 layer 얻어오기
```
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
```
- last_layer = pre_trained_model.get_layer('mixed7')
모든 layer에는 이름 존재<br>
마지막으로 사용할 layer의 이름 얻어오기<br>
convolution (7, 7)의 output<br>

### dropout
layer 추가 > dropout > 모델 생성 >  컴파일<br>

```
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)          

# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
```

### Augmentation
```
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
```

### [dropouts](https://www.youtube.com/watch?v=ARq74QuavAo)
신경망에서 임의의 수의 뉴런을 제거<br>
사용하는 이유<br>
1. 인접한 뉴런이 종종 비슷한 가중치를 갖게되어 overfitting으로 이어질 수 있으므로 무작위로 일부를 제거하면이를 제거 할 수 있다
2. 종종 뉴런이 이전 계층에 있는 뉴런의 입력 값을 초과 할 수 있으며 결과적으로 지나치게 specialization 될 수 있다

[Week 3 실습_dropouts](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)<br>


[Exercise 7 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%207%20-%20Transfer%20Learning/Exercise%207%20-%20Question.ipynb)
















