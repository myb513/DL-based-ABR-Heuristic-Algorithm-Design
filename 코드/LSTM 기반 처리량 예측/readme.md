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


