# Deep Learning 테스트
* 배경 : ES 토크나이저 및 애널라이저를 활용한 텍스트 데이터 분석의 한계 도출
* 목적 : DL 테스트를 통한 html데이터 카테고리 분류
* 분류 : 범죄(1) 범죄X(0)의 이진분류
  * test_3까지는 성범죄(1) 성범죄X(0)으로 분류했으나 양질의 데이터 부족으로 인해 위와 같이 변경
* 기간 : 2021.10.05 ~ 2021.12.03
* 사용 툴 : Google Colab, Kaggle

# Note
* DL관련 총 5번의 테스트를 진행했으며, 코드는 2차 테스트부터 게시되어 있다
* 대상 데이터는 초기 test_2, test_3 파일은 훈련데이터로 train_data를, 시험데이터로 test_data를 사용했으며
* test_4 버전부터는 EDA기법으로 늘려준 eda_data파일을 훈련데이터로 사용하였다

# 결론
* 예측 결과물은 Excel 파일로 저장되어 있으며,
* 4개의 모델이 각 최고예측률 약 68%, 84%, 81%, 98%를 보임
<img src="/uploads/c77ed7fef6413e46b4cb5e18a4c11c6d/image.png" />

# ResearchNote
* 10월 : [바로가기](researchNote/dl_oct.md)
* 11월 & 12월 : [바로가기](researchNote/dl_nov_dec.md)


# 개인적인 추후 연구방향제안
* 학습 데이터를 eda_data(**15000개**) 보다 더 크게 설정
  * 카테고리별 라벨링도 최대한 고르게 이뤄져야
* 위 사항 충족시 학습 후 모델 최대성능 옵션파일(.h5) 다운로드
* 코드를 통으로 돌리지 말고 아래 코드 사용해서 test데이터에 대한 성능 검증

Sample

    test = pd.read_csv('test_data.csv')
    input_2 = test['html']
    test_input = bert_encode(result_2,tk,max_len=150)  # max_len은 변경가능!
    bert_model = models.load_model('best-bert-model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
    print(bert_model.evaluate(test_input,pd.get_dummies))
    result = bert_model.predict(test_input)
    print(result)


* 은어가 추가된 우리만의 단어사전 커스터마이징
  * FastText / Word2Vec 등의 모듈을 활용한 단어사전 커스텀 방안 예상
