# 데이터셋 만들기
* 대상 : WinSCP의 cralwer_v4 파일들 중 일부를 랜덤하게 로드 + 기존 선행 연구 데이터파일(txt)
* 라벨링 : 범죄(1) 범죄X(0)의 이진분류
    * 초기 성범죄 / 성범죄X 로 분류했으나, 양질의 데이터 부족으로 인해 위와 같이 변경 
    * 각 html의 내용 확인 및 사이트 열람을 통한 수동분류(오류 가능성 존재)
* 모델 훈련/학습 데이터셋 만든 코드는 (.ipynb) 파일에, 최종 데이터셋 완성본은 (.csv) 파일로 저장

# 기간별 Note
* 10월 : [바로가기](./researchNote/dataset_readme.md)