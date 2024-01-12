# customer-classifier with KNN

Bu proje, bir e-ticaret şirketinin müşterilerini RFM (Recency, Frequency, Monetary) metrikleri üzerinden segmente etmek ve ardından K-Nearest Neighbors (KNN) algoritması ile bu segmentlere yeni müşterileri tahmin etmek amacıyla geliştirilmiştir.

## Veri Ön İşleme

Proje, `data_prep.py` dosyasında müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayarak RFM skorlarına dönüştürmektedir. Ardından, müşterileri belirli segmentlere atamak için RFM skorları kullanılmıştır.

## Model Eğitimi

Model eğitimi, `model_training.py` dosyasında gerçekleştirilmiştir. Bu aşamada KNN algoritması kullanılmıştır. Sayısal ve kategorik özellikler, bir `ColumnTransformer` ve `Pipeline` kullanılarak ön işleme yapılmıştır.

## Sonuçlar

Eğitilen model, test setinde yüksek doğruluk oranları elde etmiş ve müşteri segmentasyonunu başarılı bir şekilde gerçekleştirmiştir. Confusion Matrix ve Classification Report sonuçları, modelin performansını detaylı bir şekilde göstermektedir.

## Kullanım

Projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. `veri_on_isleme.py` dosyasını çalıştırarak veri ön işleme aşamasını gerçekleştirin.
2. `model_egitimi.py` dosyasını çalıştırarak modelinizi eğitin.


