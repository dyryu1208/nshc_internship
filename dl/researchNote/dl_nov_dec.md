# 목차

* [2021.11.01](#anchor-20211101)
* [2021.11.09 ~ 2021.11.12](#anchor-20211109)
* [2021.11.15](#anchor-20211115)
* [2021.11.22](#anchor-20211122)
* [2021.11.23](#anchor-20211123)
* [2021.11.24 ~ 2021.11.30](#anchor-20211124)
* [2021.12.02 ~ 2021.12.03](#anchor-20211202)


# 2021.11.01
## 모델 성능 향상관련 테스트
1. 모델 성능 향상을 위해 Embedding layer의 output_dim을 늘려주면 좋지 않을까?
    * (vocab_size는 3만개가 넘는데 output_dim이 16개에 고정되어 있으면 제대로된 학습이 안될 것 같다는 생각)
* 자료를 찾던 중 다음과 같은 레퍼런스(stackoverflow) 발견
  * 출처: <https://stackoverflow.com/questions/51328516/what-embedding-layer-output-dim-is-really-needed-for-a-dictionary-of-just-10000>

* 따라서, output_dim 층 크기를 점점 늘려가면서(256까지) 모델 성능 테스트 실행
  * a) 32 : 16이었을때와 크게 차이가 없는 것 확인
  * b) 64 : 16이나 32이었을때와 크게 다르지 않음 
  * c) 128 : RNN 모델에서는 제대로 학습 못함(임베딩 벡터가 길어져서?) / LSTM & GRU는 이전과 크게 차이없음
  * d) 256 : RNN 성능 다시 복구 / LSTM 개선 / GRU 살짝 하락
* 테스트 결과 큰 점수향상은 없음….

2. 현재 가지고 있는 훈련 데이터셋의 개수는 약 2500개 적다면 적을 수 있음!
   * 사전훈련된 워드 임베딩 사용가능(Pre-trained Word Embedding)
     * Word2Vec, FastText, GloVe 등의 모듈을 가져와서 사용가능
   * 참고자료: <https://wikidocs.net/33793> , <https://ebbnflow.tistory.com/154> , <https://wikidocs.net/22885> , <https://wikidocs.net/86083>

## 연구방향 토의
* **팀장님 코멘트 – 근본적 문제 해결요청**
  - 전처리 오류는 Tokenizer()에도 예외처리 특수문자를 표시하고, Tokenizer()에 데이터를 넣기 전에도 소수점/$/BTC/USD 앞뒤로 숫자가 붙는 것들을 하나의 토큰으로 보도록 전처리 하는 작업 필요할 듯!
  - 결국 우리가 쓰는 모델은 각 string 단어들의 배열을 보고 분류하는데, 
  - 위키 사이트에서 “마약/마켓/삽니다or팝니다/ 링크주소….” 라고 되어있는 html과 
  - 마켓 사이트에서 “마약/마켓/팝니다/연락처/링크주소……”라고 되어있는 html을 어떻게 잘 구분지을지에 대한 방안 검토
* **연구원님 코멘트**
  - DL 층 여러 개 쌓는것 추천(하나씩 추가하는 테스트 과정필요)
  - ML활용 분류방안도 적극적으로 검토


# 2021.11.09
토픽모델링 적용하던 중 근본적인 전처리 문제점 발견

    edit = " ".join(w for w in nltk.wordpunct_tokenize(sent_2) \
            if w.lower() in words or not w.isalpha())

위 로직에서 cannabis, shit, btc, bitcoin….과 같은 은어들을 필터링하고 있었음
  - 주석처리함으로써 해결


# 2021.11.11
* Notebook 파일이 아닌 VSCode와 같은 py 모듈에서도 동일한 결과를 내는 코드 작성
* 전처리 오류 관련 수정


# 2021.11.12
* BERT 기법 정보조사 시작
* Google Colab 내 TPU 사용관련 정보조사 
  * 코드 작성시 원인불명의 이유로 작동되지 않는 현상발생
  * Google Colab 이외의 툴에서는 작동하지 않는 것 확인
    * 사용 보류


# 2021.11.15
모델 학습으로 아래 옵션 적용시 GRU모델 정확도 80%달성

    model3 = Sequential()
    model3.add(Embedding(vocab_size,128,input_length=2000))
    model3.add(GRU(32,return_sequences=True))
    model3.add(Flatten())
    model3.add(Dense(1,activation='sigmoid'))
    cp = callbacks.ModelCheckpoint('best-gru-model2.h5')
    es = callbacks.EarlyStopping(patience=3,restore_best_weights=True,verbose=1)
    rl = callbacks.ReduceLROnPlateau(patience=2)
    opt = optimizers.Adam(learning_rate=0.01)
    model3.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])
    history2 = model3.fit(train_seq, y_train, epochs=3, batch_size=32, validation_split = 0.2, callbacks=[cp,es,rl])

향후 테스트를 진행하더라도 GRU 모델은 위 옵션으로 고정

# 2021.11.22
## 외부 단어 데이터(FastText) 파일을 가져와 test_4의 단어들과 비교하는 함수 만들기 + 층 추가
* Colab은 외부데이터 업로드가 너무 오래 소요
  * FastText 관련 연구는 Kaggle로 진행
* **테스트 결과**
  * RNN 정확도: 약 80% 
    * 전부 동일한 확률(51.4%, 1)로 예측했음 
    * 사실상 의미없으므로 기존 코드 유지
  * LSTM 정확도: 약 77% 
    * 다양한 확률로 예측했음 
    * 의미있는 결과 
  * 학습시간 1회 학습당 평균 50초!
  * FastText데이터에 없는 단어들로 구성된 html은 거의 다 틀리는 현상 발생

* **팀장님 피드백**
  * Kaggle말고 Colab이나 VSCode에서도 사용가능한 코드  추가
  * FastText 데이터에 우리 데이터의 은어를 녹이는 방안 탐색


# 2021.11.23
## FastText 파일을 Kaggle 이외 툴들에서 사용하는 코드 작성
VSCode : 파일을 폴더에 로드 후 메모리설정 12988(MB)로 변경

    import numpy as np
    def load_embedding(path):
        embeddings = {}
        with open(path,encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split()   
                word = values[0]                
                vector = np.asarray(values[1:],dtype=np.float32)
                embeddings[word] = vector        
        return embeddings
    embeddings=load_embedding('C:/Users/NSHC/Desktop/work/프로젝트/프로젝트2_텍스트분류/test_4/crawl-300d-2M.vec')
    # print(embeddings[‘bitcoin’])

Google Colab 

    from google.colab import files
    files.upload()

위 코드 작성하고 로컬 파일 선택하는 방법 
* 대용량 파일 로드에는 부적합(소요시간 )

구글드라이브 MyDrive에 파일 업로드 후 다음과 같은 코드 작성

    from google.colab import drive
    drive.mount('/content/gdrive/')
    def load_embedding(path):
        embeddings = {}
        with open(path) as f:
            for line in f:
                values = line.rstrip().split()  
                word = values[0]               
                vector = np.asarray(values[1:],dtype=np.float32)
                embeddings[word] = vector         
        return embeddings

    embeddings = load_embedding('/content/gdrive/MyDrive/crawl-300d-2M.vec')

매번 마운트를 위한 로그인 및 인증코드 클릭하는 불편함 존재


# 2021.11.24
* **진이준 연구원님 피드백**
  * FastText로만 학습방향을 설정하지 말고
  * 이전 코드 전처리 과정에 TF-IDF 등의 벡터화 과정을 추가해 그 결과를 FastText적용 모델과 비교할 것!
* **FastText 정보조사 요구사항**
  * FastText파일이 어떤 단어들을 내포하는지?
  * Use-Case가 있는지?
  * Vector파일의 추후 활용방안? (우리 데이터셋 단어와의 비교 또는 인사이트 작성)


# 2021.11.25
* FastText 정보조사 완료
  * 관련 문서는 word파일로 게시
* FastText를 쓰지않는 TF-IDF 기반 전처리시도
  * PorterStemmer 과정에서 램 터짐현상 및 작업지연(최소 25분) 발생
    * 작업 보류...

# 2021.11.29
## FastText에 단어를 추가하는 방안 탐색
  * 출처 : <https://velog.io/@hamdoe/fasttext-add-pretrainedVectors>
  * 프로세스 : 
    1. Vector파일 재학습
    2. 은어에 대한 벡터 찾기(유사한 단어를 기준으로)
    3. 은어 및 벡터값 append
* 테스트 결과 모델 재학습에 드는 시간이 너무 오래 걸리는 현상 발생

# 2021.11.30
* 전날 오류 해결 시도
  * multiprocessing의 cpu를 최대치로 끌어올려 실행했음에도 13시간 이상 소요되는 듯
  * 작업 보류...
  
관련 코드

    import fasttext
    model = fasttext.train_unsupervised('/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',epoch=5,dim=100,model='skipgram',
                                    thread=multiprocessing.cpu_count() -1,lr=0.05)
    words = model.get_words()
    test = model.get_word_vector("videofrench")
    print(test.shape)
    print(test)
    # fasttext 모델에서 배제된 단어추가
    oov_words = []
    data = '/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
    with open(data, 'a') as f:
        for w in oov_words:
            v = model.get_word_vector(w)
            vstr = ''
            for vi in v:
                vstr += ' ' + str(vi)
            try:
                f.write(w+vstr+'\n')
            except:
                pass


# 2021.12.02
* 기존 FastText 적용모델 성능 재점검
* FastText 이외의 방법 검토
  * BERT 모델 정보조사 재개


# 2021.12.03
## BERT 모델 코드 작성 및 테스트
## test_5.ipynb 파일에 작업
* 코드 출처 : 
  * <https://www.analyticsvidhya.com/blog/2020/10/simple-text-multi-classification-task-using-keras-bert/>
  * <https://www.kaggle.com/funxexcel/keras-bert-using-tfhub-trial>
* GPU 기준 max_len 옵션을 110보다 높게 설정하면 RAM 터지는 현상 발생
* max_len을 최대값(110)으로 설정 후 테스트
  * 학습 소요시간 : 2회 학습에 약 19 분 (회당 10분)
  * 정확도 : 약 98%
* 타 모델들(RNN,LSTM,GRU….)에 비해 높은 정확도를 보이고, 소요시간도 적은편(max_len 늘리면 학습시간은 증가할듯!)
