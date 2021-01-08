# 2. Convolutional Neural Networks in TensorFlow

## WEEK 1 : Exploring a Larger Dataset

[Kaggle의 고양이와 개](https://www.kaggle.com/c/dogs-vs-cats)들의 훨씬 더 큰 데이터셋에 적용<br>
25,000마리의 고양이, 개 이미지로 구성<br>
- 작은 데이터 집합을 사용하면 overfitting의 위험이 커짐
- 큰 데이터 집합을 사용해도 overfitting 발생하기는 함

- overfitting을 피하기 위한 다른 전략
기존 모델을 사용하고 이전 학습을 하는 것<br>

25000개 중 2,000개로 훈련, 1000개로 테스트<br>
[Week 1 실습_Dog_Cat](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=Fb1_lgobv81m)


### Zip 파일 압축풀기
```
import os
import zipfile

local_zip = '/tmp/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()
```

### 디렉토리를 변수로 설정
```
base_dir = '/tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
```

### 파이썬 목록에 로드
```
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])
```
파일 이름의 목록 볼 수 있음


### 훈련 및 검증 데이터 개수 확인
```
print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))
```

### matplotlib을 이용해 데이터 시각화해서 보기 가능

### Sequential
### compile
### train_generator, validation_generator
train_generator : 2개 클래스에 2000개 이미지<br>
validation_generator : 2개 클래스에 1000개 이미지<br>
### model.fit

### + 정확도를 높이는 방법 : 이미지 자르기

### history 객체를 사용하여 정확도 및 손실 값 확인
- 학습 정확도 및 손실
2epochs 이후 정확도는 올라가고 손실 0.75에서 평준화
> 훈련 데이터를 overfit(과하게 학습)했기에 2epochs 이후에는 훈련할 포인트가 없음

- 유효성 검사 정확도 및 손실
2epochs 이후 정확도는 내려가고 손실은 올라감

> 전체 데이터 집합을 사용하면 더 나은 결과를 얻을 수 있음







