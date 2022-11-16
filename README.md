# 기침 소리를 이용한 **COVID-19 검출 AI**

[배경과 목적]

우리나라의 경우에는 사회적 거리두기에 이어 지난 5월 2일부터 실외 마스크 착용 의무가 해제되는 등 방역조치가 해제됐었습니다.

하지만 추운 날씨에 활발한 바이러스의 특성상 올 가을부터 COVID-19의 재유행 가능성도 커 앤데믹까지는 아직 이르다는 전문가들의 의견이 있었고 다시 변이 바이러스가 유행하기 시작했습니다.

COVID-19 검출을 위한 표준 방법은 역전사 중합효소 연쇄 반응(RT-PCR) 검사입니다.

그러나 이 방법은 비용이 많이 들고 시간이 많이 걸리기 때문에 재유행으로 인해 감염자가 다시 급증하여 검사량이 몰린다면 자칫 의료체계가 무너질 가능성도 있습니다.

때문에 대규모로 배포할 수 있으며, 기존의 한계점을 해결할 수 있는 **대체 진단 도구**가 필요합니다.

[역할]  
base code 작성 , 코드 총괄

[목표]

- librosa 음성 파일 분석 프로그램을 가지고 기침소리 데이터를 딥러닝 학습을 할 수 있게 만든다.
- CNN 모델로 코로나 양성, 음성을 판별하고 정확도를 높인다.
- 판별 score 0.6 이상
- 대회 1등 ( 주어진 데이터를 가지고 가장 높은 score를 만들어내는 것 )

[진행]
2022.06.24 ~ 2022.07.08

1. **데이콘 음향 데이터 COVID-19 검출 AI 경진대회에서 데이터 다운로드**

train [Folder] : 기침소리 학습용 오디오 파일 (3805개)

```
│	├ 00001.wav : 검사자들의 기침소리

│	├ 00002.wav : 검사자들의 기침소리

│	└ ...

```

test [Folder] : 기침소리 테스트용 오디오 파일 (5732개)

- wav 확장자
- trrin 데이터에서 양성은 전체 약 10%
- 잡음과 기침이 아닌 말소리도 섞여있음
- 기침 소리 파일 길이 10초 이하

CSV파일에는 나이, 성별, 기침 여부, 발열 여부, 확진 여부 정보가 있다.

![Untitled (5)](https://user-images.githubusercontent.com/87513112/201999170-71df0e12-81c5-48c3-a349-fe27d1b8588c.png)
기침 소리로만 판별을 하고자 확진 여부 외에 사용하지 않음

2. 음향 데이터의 이해

![image](https://user-images.githubusercontent.com/87513112/202076128-8db58f31-3ed1-4aed-890d-43c693be2ed8.png)
![image2](https://user-images.githubusercontent.com/87513112/202076131-6b0f79c9-50e8-4bab-939b-3a094f105969.png)
![image3](https://user-images.githubusercontent.com/87513112/202076134-8fda92fa-f874-4d48-b129-eb6cb322136b.png)
![image4](https://user-images.githubusercontent.com/87513112/202076136-7f6f9cdb-85ea-4af2-b72a-2575234c2627.png)


3. 기침 소리 데이터 전처리

    1. 전처리 없이 있는 그대로 사용한 데이터
    2. 무음만 제거한 데이터
    3. 한 파일 당 2초 오디오 분할한 데이터
    4. 음향 데이터 이미지 변환

![image6](https://user-images.githubusercontent.com/87513112/202076140-4a429dbb-8a36-4f3f-961b-1b7ed4092753.png)
![image7](https://user-images.githubusercontent.com/87513112/202076142-f15060ab-b867-492f-b29a-10161fc03796.png)
![image8](https://user-images.githubusercontent.com/87513112/202076146-00aec008-e715-4d6c-86c1-c4479225a185.png)
![image9](https://user-images.githubusercontent.com/87513112/202076152-f73fdc5e-6ec4-45fb-86bd-cea80b064f3c.png)


- **wav 기침소리 파일들을 librosa를 이용하여 데이터 시각화**

****MFCC (Mel Frequency Cepstral Coefficient) 변환****

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
```

<img src=https://user-images.githubusercontent.com/87513112/201999826-79ae61f3-ca59-49e5-835c-344fd466123d.png width="400" height="200"/>


**mel spectrogram 변환**
![image5](https://user-images.githubusercontent.com/87513112/202076138-bf219321-069b-4e3d-8a46-cacf6029fe56.png)

```python
mel_spectrogram = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
```

<img src=https://user-images.githubusercontent.com/87513112/201999493-cf191f39-1a37-4389-8bb1-1deebf891c38.png width="300" height="200"/>




5. 모델 학습
    ![image10](https://user-images.githubusercontent.com/87513112/202076153-0b90ecc1-a338-4c35-9c12-e7f199f35925.png)
    **CNN 모델로 학습하고 코로나 확진 여부 판별**
    
    Found 3426 images belonging to 2 classes. —> negative  
    Found 379 images belonging to 2 classes. —> positive  
    Found 5732 images belonging to 1 classes. —> test  
    
    — ImageDataGenerator 데이터 학습 테스트 나눔
    —  모델 100 epoch 학습
    
    — confusion matrix 확인
    
    — submission csv에 예측 값 변경 후 데이콘에 제출하여 점수 확인



[결과]

원본, 무음처리, 분할 오디오 학습한 CNN 모델 정확도 최대 0.52

[개선, 느낀 점]

1. 코로나 기침소리 학습 데이터 약 3000개 외에는 데이터가 없다. (데이터 서치를 해보았지만)
데이터 부족으로 최대 정확도 이상 변화를 줄 수 없음
2. 확실히 대회 최대 점수도 발열 여부 등 추가 데이터와 함께 학습함에도 불구하고 0.62 이상을 넘기지 못하는 것을 보니 데이터 부족이 원인인 것을 알 수 있었다.
3. 음향 데이터에 대한 지식이 필요하다.








### 파일 구조

```markdown
```
covid_19   
├─ open   
│  ├─ train   
│  ├─ test   
│  ├─ unlabeled   
│  ├─ mute_removal_audio   
│  ├─ cropped_audio   
│  ├─ train_data.csv   
│  └─ test_data.csv   
│   
├─ mel   
├─ mfcc   
│   
└─ deep_learning.ipynb   
```
```

### open

- train
음성, 양성 기침소리 훈련 데이터

- test
음성, 양성 기침소리 테스트 데이터

- unlabeled
음성, 양성 구분할 수 없는 unlabeled 데이터

- mute_removal_audio
10 데시벨 이하의 소리 구간은 잘라낸 wav 오디오 데이터

- cropped_audio
pydub를 이용한 wav 오디오 분할 데이터 (데이터를 쪼개서 많이)

- train_data.csv
train_data_dataframe
id / age / gender / respiratory_condition / fever_or_muscle_pain / covid19(1 양성, 0 음성)

- test_data.csv
test_data_dataframe
id / age / gender / respiratory_condition / fever_or_muscle_pain

### mel




- train
negative, positive 나누어서 2개의 class로 저장
- test
1개의 class로 저장

### mfcc




- train
negative, positive 나누어서 2개의 class로 저장
- test
1개의 class로 저장

### deep_learning.ipynb

- 파일 경로
딥러닝 학습에 사용할 [id, 확진 여부, 파일 경로들을 datafream으로 만들어 사용

- 오디오 전처리
— 묵음 처리
— pydub 오디오 분할
- 이미지 생성
— mel_spectrogram_extraction / mel_spectrogram 이미지 생성
— mfcc_extraction / mfcc 이미지 생성
- 모델 학습
— tensorflow GPU 활성화
— ImageDataGenerator 데이터 학습 테스트 나누기
—  모델 학습
    
    ```
    model = tf.keras.models.Sequential([
        #first_convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        #second_convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        #third_convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        #fourth_convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax') 
    ]) 
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    ```
    
    — confusion matrix 확인
    
    ```
    accuracy = loaded_model.evaluate(test_generator)
    print('n', 'Test_Accuracy: ', accuracy[1])
    pred = loaded_model.predict(test_generator)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(pred, axis=1)
    print('confusion matrix')
    print(confusion_matrix(y_true, y_pred))
        #confusion matrix
    f, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()
    ```
    
    — submission csv에 예측 값 변경 후 데이콘에 제출하여 점수 확인
