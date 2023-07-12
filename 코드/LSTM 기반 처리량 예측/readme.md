## Loss 줄이기 대작전

### 5종류 파일별 평균 loss, RMSE (0710 - 오전 11시)

```
Average Loss for norway_bus: 0.8200654978661434
Average RMSE for norway_bus: 2.8550689591740683

Average Loss for norway_car: 0.8632435977148513
Average RMSE for norway_car: 1.6125853899249403

Average Loss for norway_ferry: 1.050452275481075
Average RMSE for norway_ferry: 1.671742215196391

Average Loss for norway_metro: 0.9014781301841139
Average RMSE for norway_metro: 1.3037957097508026

Average Loss for norway_train: 0.7231793492766363
Average RMSE for norway_train: 1.8218095248378325


Average Loss for norway_tram: 1.0874163205236462
Average RMSE for norway_tram: 0.8635825068434021
```

### 5종류 파일별 평균 loss, RMSE (0710 - 오후 12시)
#### timestep 2배 늘리고, learning rate 0.005로 설정
```
Average Loss for norway_bus: 0.45112204260152317
Average RMSE for norway_bus: 2.4936502817843755

Average Loss for norway_car: 0.5429791019608577
Average RMSE for norway_car: 1.1029296274843128

Average Loss for norway_ferry: 0.586785692954436
Average RMSE for norway_ferry: 1.3564371347041644

Average Loss for norway_metro: 0.5291992735117674
Average RMSE for norway_metro: 1.0041247437768004

Average Loss for norway_train: 0.44004547343190226
Average RMSE for norway_train: 1.6507045486163885

Average Loss for norway_tram: 0.624087049692337
Average RMSE for norway_tram: 0.6825757067016001
```

### 5종류 파일별 평균 loss, RMSE (0711 - 오후 2시)
#### 모델 층 구조 추가 32개짜리 레이어?
```
Average Loss for norway_bus: 0.41885597238560085
Average RMSE for norway_bus: 2.432128062731251

Average Loss for norway_car: 0.5383156654424965
Average RMSE for norway_car: 1.0910070509313299

Average Loss for norway_ferry: 0.5598084934987128
Average RMSE for norway_ferry: 1.3276279754051459

Average Loss for norway_metro: 0.4829714709892869
Average RMSE for norway_metro: 0.9668349026835052

Average Loss for norway_train: 0.4275669992147457
Average RMSE for norway_train: 1.6288726327126775

Average Loss for norway_tram: 0.6039361167931929
Average RMSE for norway_tram: 0.6506575575299729
```

### 5종류 파일별 평균 loss, RMSE (0711 - 오후 3시)
#### epoch 10-> 100 으로 변경 


>epoch 100으로 바로 갔더니 ,, 오버피팅 된듯.
>10~ 50사이를 10 간격으로 모두 해보고 제일 나은거 ㄲ
>

```
Average Loss for norway_bus: 2.3307771787715508
Average RMSE for norway_bus: 2.339673771719343

Average Loss for norway_car: 3.6836642963656536
Average RMSE for norway_car: 1.1960164299810598

Average Loss for norway_ferry: 3.060195946507156
Average RMSE for norway_ferry: 1.4451066300335025

Average Loss for norway_metro: 3.2672178003937007
Average RMSE for norway_metro: 1.0813373015585048

Average Loss for norway_train: 2.5784553364141
Average RMSE for norway_train: 1.8624297616794598

Average Loss for norway_tram: 3.5196363804695596
Average RMSE for norway_tram: 0.6599491557924633
```

### 5종류 파일별, epoch 별, avg(RMSE) (0711 - 오후 3시)
#### 가장 정확도가 높은 epoch 값을 추적하는 그래프를 그려보자

<div class="user-image">
        <img src="시각자료/epoch별 정확도 (10~50).png" alt="" />
</div>

-> 그래프를 보면 10~20 epoch로 하면 될 것 같다는 생각을 함. why? 합리적으로 rmse 값이 전체적으로 낮은 부분이니깐!

### 서로다른 층 구조 모델 3개 비교 loss, RMSE (0711 - 오후 5시)
#### 시각화 코드 에러 이슈로 콘솔출력 값 수기 입력
```
# 모델 1
Average Loss for norway_bus: 0.4277389136993367
Average RMSE for norway_bus: 2.483556167239904

# 모델 2
Average Loss for norway_bus: 0.423200525464895
Average RMSE for norway_bus: 2.3877440547929756

# 모델 3
Average Loss for norway_bus: 0.44199857407290005
Average RMSE for norway_bus: 2.3445965110316576

# 모델 1
Average Loss for norway_car: 0.5266553272182742
Average RMSE for norway_car: 1.1137567739190306

# 모델 2
Average Loss for norway_car: 0.5419865385629237
Average RMSE for norway_car: 1.065013561803757

# 모델 3
Average Loss for norway_car: 0.5278499145060778
Average RMSE for norway_car: 1.2076046218577345

# 모델 1
Average Loss for norway_ferry: 0.5660448386799544
Average RMSE for norway_ferry: 1.3476696091517664

# 모델 2
Average Loss for norway_ferry: 0.547693053074181
Average RMSE for norway_ferry: 1.3306689245935162

# 모델 3
Average Loss for norway_ferry: 0.541606237925589
Average RMSE for norway_ferry: 1.2782834034212676

# 모델 1
Average Loss for norway_metro: 0.4791893146932125
Average RMSE for norway_metro: 0.9578108050957672

# 모델 2
Average Loss for norway_metro: 0.4765708018094301
Average RMSE for norway_metro: 0.9282643189992317

# 모델 3
Average Loss for norway_metro: 0.4830556891858578
Average RMSE for norway_metro: 0.9699208809326457

# 모델 1
Average Loss for norway_train: 0.4292234616309759
Average RMSE for norway_train: 1.610844508434629

# 모델 2
Average Loss for norway_train: 0.43025303889243377
Average RMSE for norway_train: 1.6481098579220819

# 모델 3
Average Loss for norway_train: 0.41436857725715354
Average RMSE for norway_train: 1.610064776048801

# 모델 1
Average Loss for norway_tram: 0.590874460159934
Average RMSE for norway_tram: 0.6413066036774696

# 모델 2
Average Loss for norway_tram: 0.598417838885715
Average RMSE for norway_tram: 0.6486776374131601[Uploading norway_train_merge_result.txt…]()


# 모델 3
Average Loss for norway_tram: 0.5909878046784017
Average RMSE for norway_tram: 0.6345559510082625
```

### 5종류 파일별, epoch 별, avg(RMSE) (0712 - 오전 10시)
#### 가장 정확도가 높은 epoch 값을 추적하는 그래프를 그려보자

<div class="user-image">
        <img src="시각자료/epoch별 정확도 (100~500).png" alt="" />
</div>

### 같은 종류의 파일들을 병합하여 많은 양의 데이터를 학습시켜보자 (0712 - 오전 12시)

```
Average Loss for norway_bus: 0.48099879175424576
Average RMSE for norway_bus: 2.5172940979055785

Average Loss for norway_car: 0.3148403521627188
Average RMSE for norway_car: 1.376522023474511

Average Loss for norway_ferry: 0.1533771799877286
Average RMSE for norway_ferry: 1.5898911879670978

Average Loss for norway_metro: 0.4496259056031704
Average RMSE for norway_metro: 0.953003822698219

Average Loss for norway_train: 0.1367283184081316
Average RMSE for norway_train: 1.560153460898116

Average Loss for norway_tram: 0.17534580454230309
Average RMSE for norway_tram: 0.6122976128095053
```


### 서로다른 층 구조 모델 3개 비교 loss, RMSE (0711 - 오후 5시)
#### 시각화 코드 에러 이슈로 콘솔출력 값 수기 입력
#### 2번 모델은 fully connected 층 번갈아가며 쌓음 
```
# 모델 1
Average Loss for norway_bus: 0.4277389136993367
Average RMSE for norway_bus: 2.483556167239904

Average Loss for norway_bus: 0.41329867036446283
Average RMSE for norway_bus: 2.4493275211208436

# **모델 2**
Average Loss for norway_bus: 0.423200525464895
Average RMSE for norway_bus: 2.3877440547929756
↓
Average Loss for norway_bus: 0.4158209310439618
Average RMSE for norway_bus: 2.3758773997697857

# 모델 3
Average Loss for norway_bus: 0.44199857407290005
Average RMSE for norway_bus: 2.3445965110316576

Average Loss for norway_bus: 0.4214636764853545
Average RMSE for norway_bus: 2.3459457710704426

# 모델 1
Average Loss for norway_car: 0.5266553272182742
Average RMSE for norway_car: 1.1137567739190306

Average Loss for norway_car: 0.537993042729795
Average RMSE for norway_car: 1.1149703135520552

# **모델 2**
Average Loss for norway_car: 0.5419865385629237
Average RMSE for norway_car: 1.065013561803757
↓
Average Loss for norway_car: 0.515500831262519
Average RMSE for norway_car: 1.089295040324735

# 모델 3
Average Loss for norway_car: 0.5278499145060778
Average RMSE for norway_car: 1.2076046218577345

Average Loss for norway_car: 0.532243927475065
Average RMSE for norway_car: 1.101109357948759

# 모델 1
Average Loss for norway_ferry: 0.5660448386799544
Average RMSE for norway_ferry: 1.3476696091517664

Average Loss for norway_ferry: 0.5780884129926562
Average RMSE for norway_ferry: 1.3517955546455371

# **모델 2**
Average Loss for norway_ferry: 0.547693053074181
Average RMSE for norway_ferry: 1.3306689245935162
↓
Average Loss for norway_ferry: 0.5230336480308324
Average RMSE for norway_ferry: 1.2412798224211539

# 모델 3
Average Loss for norway_ferry: 0.541606237925589
Average RMSE for norway_ferry: 1.2782834034212676

Average Loss for norway_ferry: 0.5714752587489784
Average RMSE for norway_ferry: 1.4190226079039763

# 모델 1
Average Loss for norway_metro: 0.4791893146932125
Average RMSE for norway_metro: 0.9578108050957672

Average Loss for norway_metro: 0.4964000040665269
Average RMSE for norway_metro: 1.0178109963553026

# **모델 2**
Average Loss for norway_metro: 0.4765708018094301
Average RMSE for norway_metro: 0.9282643189992317
↓
Average Loss for norway_metro: 0.46179626174271104
Average RMSE for norway_metro: 0.8640390700717593

# 모델 3
Average Loss for norway_metro: 0.4830556891858578
Average RMSE for norway_metro: 0.9699208809326457

Average Loss for norway_metro: 0.485748521797359
Average RMSE for norway_metro: 0.9931346681114335

# 모델 1
Average Loss for norway_train: 0.4292234616309759
Average RMSE for norway_train: 1.610844508434629

Average Loss for norway_train: 0.42492537901160266
Average RMSE for norway_train: 1.5799662286803668

# **모델 2**
Average Loss for norway_train: 0.43025303889243377
Average RMSE for norway_train: 1.6481098579220819
↓
Average Loss for norway_train: 0.421641770417669
Average RMSE for norway_train: 1.6100382217767044

# 모델 3
Average Loss for norway_train: 0.41436857725715354
Average RMSE for norway_train: 1.610064776048801

Average Loss for norway_train: 0.4109073242704783
Average RMSE for norway_train: 1.7172948403106445

# 모델 1
Average Loss for norway_tram: 0.590874460159934
Average RMSE for norway_tram: 0.6413066036774696

Average Loss for norway_tram: 0.5994756830457065
Average RMSE for norway_tram: 0.6490862435053213

# **모델 2**
Average Loss for norway_tram: 0.598417838885715
Average RMSE for norway_tram: 0.6486776374131601
↓
Average Loss for norway_tram: 0.5756453441322914
Average RMSE for norway_tram: 0.6241858856390182

# 모델 3
Average Loss for norway_tram: 0.5909878046784017
Average RMSE for norway_tram: 0.6345559510082625

Average Loss for norway_tram: 0.5983663704911513
Average RMSE for norway_tram: 0.6304184990989777
```

