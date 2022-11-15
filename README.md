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

COVID-19의 두드러진 증상은 기침과 호흡 곤란을 포함합니다.

**AI 기술을 활용하여 기침 소리로부터 COVID-19에 대한 유용한 통찰력**을 얻을 수 있다면, 
새로운 진단 도구의 설계도 충분히 가능 할 수 있다고 예상합니다.

대회 1등 ( 주어진 데이터를 가지고 가장 높은 score를 만들어내는 것 )

score 0.6 이상

[진행]

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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/924c12a0-6804-4f3d-a76a-225c5a1dbdbb/Untitled.png)

딥러닝으로만 문제를 해결하고자 확진 여부 외에 사용하지 않음

2. 기침 소리 데이터 전처리

1. 전처리 없이 있는 그대로 사용한 데이터
2. 직접 3000개 이상의 오디오 파일을 직접 듣고 모든 잡음 포함, 말소리 오디오를 삭제한 데이터
3. 무음만 제거한 데이터
4. 한 파일 당 2초 오디오 분할한 데이터

3. 음향 데이터 이미지 변환

- **wav 기침소리 파일들을 librosa를 이용하여 데이터 시각화**

****MFCC (Mel Frequency Cepstral Coefficient) 변환****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e003f46-fd18-4ee9-b777-6b861387a0ac/Untitled.png)

**mel spectrogram 변환**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6f35b480-00f2-42af-aec1-482350c6eff9/Untitled.png)

```python
# mel
frame_length = 0.025
frame_stride = 0.010

def feature_extraction(path):
    # mel-spectrogram
    y, sr = librosa.load(path, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
```

1. 모델 학습

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78cb7c96-e4f9-421e-86a8-466bf162cd7e/Untitled.png)

**기침 소리 데이터를 변형하여 음성, 양성 데이터를 CNN 모델로 학습하고 코로나 확진 여부 판별**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02360634-bcdb-46a4-b59e-064bc575ca06/Untitled.png)

[결과]

오디오 파일만 가지고 학습한 CNN 모델 정확도 최대 0.52

기침여부 성별 등 Dataframe 데이터도 합한 베이스라인에서 수정한 모델 0.58

하지만 기침 소리만 가지고 판별하는 것이 목표기 때문에 딥러닝으로 진행

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

![Untitled (2).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/91943566-315d-48a7-acfc-0b2d30cafa54/Untitled_(2).png)

- train
negative, positive 나누어서 2개의 class로 저장
- test
1개의 class로 저장

### mfcc

![201464885-d8b094e5-b8ae-4347-80b7-c7d4123eb818.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/19767758-924e-4225-9975-73f8996708da/201464885-d8b094e5-b8ae-4347-80b7-c7d4123eb818.png)

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
