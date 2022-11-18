# EDA(Exploratory Data Augmentation)
* 배경 : ML, DL(test_2,test_3) 테스트 결과 모델 예측률이 많이 떨어짐
* 목적 : 모델 학습에 필요한 충분한 데이터 구축
* 기간 : 2021.11.05


## Text Data Augmentation 기법
* 1) Synonym Replacement(동의어 대체) : 각 단어들의 동의어를 wordnet에서 추출한 뒤 삽입
* 2) Random Insertion(랜덤 삽입) : 문장 단어 중간중간에 랜덤한 단어 삽입
* 3) Random Deletion(랜덤 삭제) : 각 html 사이 단어들을 랜덤하게 삭제
* 4) Random Swap(랜덤 변동) : 단어들의 위치를 랜덤하게 swap

* 출처 : <https://github.com/catSirup/KorEDA/blob/master/eda.py>


## 출처에서 안전히 EDA하려면 3,4 방법을 사용하고
## 더 많이 필요하다 싶으면 1,2 까지 수행 후 사람이 수동검수하는 작업이 필요하다고 밝힘
* 일단 4) Random Swap 기법만 적용해서 데이터셋 크기 6배로 늘림(**약 15000개**)
* 결과물 : eda_data.csv