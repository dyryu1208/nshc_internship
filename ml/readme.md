# Machine Learning 테스트(2021.11.02 ~ 2021.11.04)
* 배경 : **2차 테스트**, **3차 테스트**의 결과 DL 모델 예측률이 50%대에 그침
* 목적 : ML 테스트를 통한 DL모델 예측률과의 비교
    * ML 예측률이 더 높다면 ML모델 활용방안 제고
* 대상 : train_data.csv, test_data.csv
* 코드 베이스라인 출처 : <https://www.kaggle.com/ayu1391994/nlp-using-random-forest>

# 목차
* [CountVectorizer](#1-countvectorizer)
* [CV+SVD](#2-cvsvd)
* [CatBoost](#3-catboost)
* [TF-IDF](#4-tf-idf)
* [결론](#결론)


# 1. CountVectorizer
* 관련 코드 : [바로가기](./CountVectorizer.ipynb)
* PorterStemmer 및 Countvectorizer 를 사용하여 전처리
* 이후 RandomForest 모델링
* train_data 에서는 91%의 정확도를 보였으나 test_data에서는 20%의 정확도...


# 2. CV+SVD
* 관련 코드 : [바로가기](./CV+SVD.ipynb)
* 위 CV의 다른 버전
* CountVectorize 해준 값을 TruncatedSVD 모듈로 차원축소
  * 이후 RandomForest 모델링
* train_data 에서는 91%의 정확도를 보였으나 test_data에서는 20%의 정확도...


## 2-1. Train/Test 데이터셋을 합쳐서 cv+svd 전처리 후 test데이터에 예측하는 식으로 적용
### 이 방법은 Data Leakage의 우려가 있으므로 사용하지 말 것!


# 3. CatBoost
* 관련 코드 : [바로가기](CatBoost.ipynb)
* 1번, 2번 방법으로 전처리 후 CatBoost 방식으로 모델링
  * 평균적으로 RandomForest 모델링보다 분류 문제에 최적화된 모델
* 눈에 띄는 정확도 향상 X
  

# 4. TF-IDF
* 관련 코드 : [바로가기](TF-IDF.ipynb)
* TF-IDF 기법으로 전처리 후 RandomForest 모델링
* 이 방법부터 EDA 데이터셋 사용
  * Colab으로 테스트 시 시간이 30분이상 소요되거나 RAM 터지는 현상 발생
* 기존 train_data로 진행
  * 모델링 결과 전부 0으로 답을 찍어버리는 현상 발생
  * 의미 없는 결과 도출


# 결론
* 위 4가지 방식으로 ML 기법 적용 결과 train_data의 데이터 수가 모델학습엔 아직 적다고 판단
* 데이터를 늘릴 방법 검토
  * [text_eda 기법으로 이동](/eda)