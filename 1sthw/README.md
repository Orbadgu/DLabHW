
## Introduction
Bu projenin amacı, Python'da Nesne Yönelimli Programlama prensipleri kullanılarak sıfırdan bir Multi-Layer Perceptron yapay sinir ağı geliştirmektir. Çalışmada Kaggle üzerinden alınan "Mall Customer Segmentation Data" veri seti kullanılmıştır. Orijinal veri seti unsupervised kümeleme algoritmaları için tasarlanmış olsa da, feature engineering yapılarak binary classification problemine dönüştürülmüştür. Temel hedef, müşterilerin yaş, cinsiyet ve yıllık gelir verilerine dayanarak "Yüksek Harcama Yapan Müşteri" (Harcama Skoru $\geq$ 50) olup olmadıklarını tahmin etmektir. Çalışma, veri ön işleme, model seçimi, overfitting önleme ve scikit-learn kütüphanesi ile performans karşılaştırması adımlarını içermektedir.

## Methods
**Veri Ön İşleme:**
Veri seti `pandas` kütüphanesi ile yüklenmiştir. Tahmin edici değeri olmayan `CustomerID` sütunu veri setinden çıkarılmıştır. Kategorik bir değişken olan `Gender` sütunu sayısal formata çevrilmiştir (Erkek=1, Kadın=0). Hedef değişken (`Target`), orijinal `Spending Score` 50 ve üzeri olanlar için `1`, altındakiler için `0` olacak şekilde yeniden oluşturulmuştur. Veri seti; Eğitim (%60), Doğrulama/Dev (%20) ve Test (%20) olmak üzere üç parçaya ayrılmıştır. Sürekli değişkenler `StandardScaler` kullanılarak Z-skoru normalizasyonu ile standartlaştırılmıştır.

**Model Mimarisi ve İmplementasyon:**
Tek gizli katmanlı (1-hidden-layer) bir MLP modeli, matris işlemleri için sadece NumPy kullanılarak sıfırdan oluşturulmuştur.
* **Encapsulation:** Algoritma, içerisinde yapıcı metot (`__init__`), gizli metotlar (`_forward_propagation`, `_backpropagation`, `_compute_cost`) ve açık metotlar (`fit`, `predict`) barındıran bir `CustomMLP` sınıfı olarak tasarlanmıştır.
* **Hiperparametreler:** Model seçimi için farklı gizli katman nöron sayıları (4, 8, 16) ve epoch/adım sayıları (500, 1000, 2000) kullanılarak Grid Search yapılmıştır.learning rate 0.5 olarak belirlenmiş ve Optimizasyon algoritması olarak Stokastik Gradyan İniş  mantığı tüm veri seti üzerinde uygulanmıştır.
* **Aktivasyon ve Kayıp Fonksiyonu:** Hem gizli katmanda hem de çıkış katmanında Sigmoid aktivasyon fonksiyonu kullanılmıştır. Loss function olarak Binary Cross-Entropy tercih edilmiştir.

**Model Seçimi:**
Modeller eğitim seti üzerinde eğitilmiş ve Doğrulama seti üzerinde değerlendirilmiştir. Nihai model, dev setindeki en yüksek doğruluk (accuracy) oranına göre seçilmiş; eşitlik durumunda hesaplama verimliliğini artırmak ve early overfitting önlemek adına en düşük eğitim adımına sahip model tercih edilmiştir.

## Results
Yapılan Grid Search sonucunda bu sınıflandırma problemi için en uygun hiperparametreler şu şekilde bulunmuştur:
* **Gizli Katman Nöron Sayısı (Hidden Size):** 4
* **Adım Sayısı (n_steps):** 1000

Modelin daha önce hiç görmediği **Test Seti** üzerindeki performansı:
* **Accuracy:** %72.50
* **Precision:** %73.68
* **Recall:** %70.00
* **F1-Score:** %71.79

Elde edilen Karmaşıklık Matrisine (Confusion Matrix) göre modelin her iki sınıfı da (15'e 14 şeklinde) oldukça dengeli bir şekilde tahmin ettiği görülmektedir.

Eğitim ve Doğrulama kayıplarını  gösteren learning curve incelendiğinde, girdi özelliğinin fazla (9) ve veri setinin küçük olmasından kaynaklı oluşabilecek overfitting probleminin **L2 Regülarizasyonu** sayesinde kontrol altına alındığı gözlemlenmiştir.

Aynı hiperparametreler (Logistic aktivasyon, SGD çözücü, 1000 epoch) kullanılarak Scikit-Learn'ün `MLPClassifier` modeli ile yapılan karşılaştırmada, sıfırdan yazılan OOP modelin test setinde birebir aynı performansı (%72.50 doğruluk) sergilediği görülmüş; bu da kurulan ileri/geri yayılım (forward & backpropagation) ve regülarizasyon matematiğinin doğruluğunu kesin olarak kanıtlamıştır.

## Discussion
Çalışma sürecinde modelin doğruluğunu maksimize etmek amacıyla `PolynomialFeatures` gibi özellik mühendisliği teknikleri uygulanarak girdi uzayı 3'ten 9'a çıkarılmıştır. Elde edilen bulgular, özellik sayısının artırılmasının sınırlı boyuttaki eğitim veri setlerinde (~120 kayıt) yüksek varyans riskini tetiklediğini ampirik olarak göstermiştir.

* **Regülarizasyon ve Optimizasyon:** Curse of Dimensionality ve ağın ezberleme problemini çözmek amacıyla modele **L2 Regülarizasyonu** entegre edilmiştir. Bu teknik sayesinde model, karmaşık girdi özellikleri içinde kaybolmadan en ideal decision boundaries çizebilmiş ve Scikit-Learn kütüphanesi ile eşdeğer stabiliteye ulaşmıştır. En iyi performansın Occam'ın Usturası prensibine uygun olarak sade bir mimaride (4 nöron) elde edilmesi, küçük veri setlerinde düşük model kapasitesinin genelleme yeteneğini artırdığını doğrulamaktadır.
* **Kısıtlar ve Gelecek Çalışmalar:** AVM müşteri verisindeki "Orta Gelir - Orta Harcama" kümesi, hedef değişken olarak belirlediğimiz 40-60 aralığında yoğunlaşmaktadır. Bu data overlap)sebebiyle, demografik özellikleri birebir aynı olan müşteriler farklı sınıflara düşebilmektedir. Modelin test setinde %85-90 gibi doğruluk oranlarına ulaşmasını engelleyen temel faktör bu doğrusal olmayan  karmaşıklıktır. Gelecek çalışmalarda, derin öğrenme algoritmalarının tam potansiyelini gösterebilmesi için veri seti hacmi artırılabilir veya girdi verilerini birbirinden daha net ayıran farklı clustering destekli özellik mühendisliği yöntemleri denenebilir.
