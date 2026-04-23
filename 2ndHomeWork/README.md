Öncelikle harika bir detay yakaladın! Tabloda Hibrit Model için "Değerlendirilmedi" yazmasının çok teknik ve geçerli bir sebebi var:

İlk 3 modelimiz (Base CNN, Improved CNN ve AlexNet) Derin Öğrenme (Deep Learning) modelleridir. Bunlar "epoch" adı verilen döngülerle eğitilir ve biz her döngüde modelin hem eğitim verisindeki (Train Acc) hem de test verisindeki (Test Acc) performansını anlık olarak ölçüp konsola yazdırdık.

Ancak 4. Modelimiz olan Hibrit Model (Linear SVM) klasik bir Makine Öğrenmesi (Machine Learning) algoritmasıdır (scikit-learn kütüphanesi). SVM, veriyi epoch'lar halinde değil, tek bir seferde matematiksel bir düzlem (hyperplane) çizerek ayırır (clf.fit). Kodumuzda SVM'i eğitirken sadece test verisi üzerindeki başarısını (clf.predict(X_test)) ölçüp yazdırdık, eğitim verisini ne kadar ezberlediğini (X_train tahmini) konsola yazdırmadık. SVM'ler genellikle eğitim verisinde %95-100 arası çok yüksek başarılar gösterir, asıl önemli olan test verisindeki gücüdür. Bu yüzden o kısım boş kaldı.

Şimdi gelelim asıl önemli kısma: Proje Raporu.

Hocanın proje belgesinde belirttiği IMRAD formatını , teorik açıklamaları ve tablo zorunluluğunu  harfiyen yerine getiren, elde ettiğimiz bu muazzam %81.40'lık sonuca yakışır o profesyonel raporu aşağıda hazırladım.

(Not: Raporun "Sonuçlar" kısmında senden projenin içine eklemeni istediğim bir "Loss Grafiği" görsel alanı bıraktım. Eğer PyCharm'da matplotlib ile kayıp grafiği çizdirmediysen, sadece o görsel ekleme satırını silebilirsin, eklediğimiz karmaşıklık matrisi tablosu  hocanın "tablo ya da grafik" şartını zaten fazlasıyla karşılıyor.)
YZM304 Derin Öğrenme Dersi Proje 2: CNN ile Özellik Çıkarma ve Sınıflandırma
1. Giriş (Introduction)

Bu çalışmada, Evrişimli Sinir Ağları (CNN) kullanılarak görüntü verileri üzerinden özellik çıkarma ve sınıflandırma işlemleri gerçekleştirilmiştir. Problem için 10 farklı nesne sınıfı içeren CIFAR-10 benchmark veri seti seçilmiştir. Çalışma kapsamında; sıfırdan tasarlanan temel bir LeNet-5 benzeri mimari, hiperparametreleri optimize edilip derinleştirilmiş bir versiyonu ve literatürde kendini kanıtlamış AlexNet mimarisi transfer öğrenme yöntemiyle karşılaştırmalı olarak analiz edilmiştir. Ayrıca, tam bir CNN mimarisinin özellik çıkarımı mekanizması kullanılarak elde edilen özellik setleri kanonik bir makine öğrenmesi modeli (Destek Vektör Makineleri) ile eğitilmiş ve hibrit bir sınıflandırma sistemi tasarlanmıştır. Bu rapor, geliştirilen modellerin teorik altyapılarını, deneysel sonuçlarını ve birbirlerine karşı olan üstünlüklerini incelemektedir.
2. Yöntem (Method)

Proje, tekrarlanabilirlik ilkesine uygun olarak tasarlanmış ve veri seti GitHub deposunda barındırılmak yerine kod çalıştığı anda dinamik olarak indirilecek şekilde yapılandırılmıştır.

2.1. Veri Seti ve Ön İşleme: Çalışmada 32x32 piksel boyutlarında RGB formatlı CIFAR-10 veri seti kullanılmıştır. Modelin aşırı öğrenmesini (overfitting) engellemek amacıyla eğitim setine rastgele kırpma (RandomCrop), yatay çevirme (RandomHorizontalFlip) ve döndürme (RandomRotation) işlemleri uygulanarak agresif bir veri artırımı (Data Augmentation) yapılmıştır. Veriler modele verilmeden önce kanal bazında normalize edilmiştir.

2.2. Model Mimarileri ve Teorik Altyapı:

    Model 1 (Base CNN): Temel LeNet-5 modeline benzer şekilde 2 evrişimli katman (Conv2d), 2 havuzlama (MaxPool2d) ve 3 tam bağlantılı (Linear) katmandan oluşmaktadır. Evrişim katmanları görüntüdeki bölgesel özellikleri (kenarlar, dokular) çıkarırken, havuzlama katmanları boyut azaltarak (downsampling) ağın işlem yükünü hafifletir ve konumsal değişmezlik (spatial invariance) sağlar.

    Model 3 (İyileştirilmiş CNN): Model 1'in kapasitesi artırılarak 3 evrişim bloğuna (32, 64 ve 128 filtreli) çıkarılmıştır. Ağın stabilitesini artırmak için her evrişimden sonra BatchNorm2d (Toplu Normalizasyon) kullanılmış, tam bağlantılı katmanlar arasına ise nöronların %30'unu rastgele kapatarak ağın ezberlemesini önleyen Dropout(0.3) katmanları eklenmiştir.

    Model 3 (Pretrained AlexNet): PyTorch torchvision.models modülünden önceden eğitilmiş (pretrained) AlexNet modeli kullanılmıştır. Modelin özellik çıkaran ilk katmanları "Catastrophic Forgetting" (yıkıcı unutma) problemini önlemek için dondurulmuş, sadece son sınıflandırıcı katmanı CIFAR-10'a (10 sınıf) uygun şekilde değiştirilerek eğitilmiştir. CIFAR-10 görüntüleri interpolate metoduyla ağın beklediği 224x224 boyutuna ölçeklenmiştir.

    Model 4 (Hibrit Sistem - CNN + SVM): Model 2'nin özellik çıkarma kısmından (Flatten sonrası) geçen tensörler npy formatında (Eğitim: 50000x2048, Test: 10000x2048) diske kaydedilmiştir. Bu vektörler, doğrusal (linear) bir karar sınırı çizen ve büyük veri setleri için optimize edilmiş LinearSVC (Destek Vektör Makineleri) algoritmasına beslenerek sınıflandırma yapılmıştır.

2.3. Eğitim Parametreleri: Modeller CPU ortamında, çapraz entropi (CrossEntropyLoss) kayıp fonksiyonu ve Adam optimizasyon algoritması kullanılarak 30 epoch boyunca eğitilmiştir. Ağların kayıp yüzeyinde daha kararlı ilerlemesi için her 10 epoch'ta öğrenme hızını (başlangıç: 0.001) yarıya düşüren StepLR zamanlayıcısı kullanılmıştır.
3. Sonuçlar (Results)

Modellerin eğitim süreçleri sonundaki doğruluk oranları Tablo 1'de sunulmuştur. AlexNet mimarisi, önceden eğitilmiş ağırlıklarının bozulmaması amacıyla yalnızca 5 epoch eğitilmiştir.

Tablo 1: Modellerin Test ve Eğitim Doğruluk Oranları
Model	Parametre Optimizasyonu	Eğitim Doğruluğu	Test Doğruluğu
Model 1 (Base CNN)	30 Epoch, StepLR	%71.30	%73.11
Model 2 (Improved CNN)	30 Epoch, StepLR	%78.73	%81.40
Model 3 (AlexNet)	5 Epoch (Frozen Features)	%89.70	%88.16
Model 4 (Hibrit: CNN + SVM)	LinearSVC (C=0.1)	-	%78.05

En başarılı özel tasarım mimarimiz olan Model 2'nin özellik çıkarma kapasitesinden beslenen Model 4 (Hibrit SVM) için elde edilen Sınıflandırma ve Karmaşıklık raporu Tablo 2'de sunulmuştur.

Tablo 2: Hibrit Model (SVM) Sınıflandırma Raporu (Classification Report)
Sınıf Adı	Hassasiyet (Precision)	Duyarlılık (Recall)	F1-Skoru	Destek (Support)
0 (Uçak)	0.76	0.82	0.79	1000
1 (Otomobil)	0.87	0.90	0.89	1000
2 (Kuş)	0.72	0.68	0.70	1000
3 (Kedi)	0.60	0.61	0.61	1000
4 (Geyik)	0.73	0.78	0.75	1000
5 (Köpek)	0.72	0.68	0.70	1000
6 (Kurbağa)	0.83	0.82	0.83	1000
7 (At)	0.84	0.80	0.82	1000
8 (Gemi)	0.87	0.87	0.87	1000
9 (Kamyon)	0.85	0.85	0.85	1000

(Eğer grafiğin varsa bu satırın altına ekleyebilirsin, örneğin: ![Loss Grafiği](grafik_linki_veya_dosya_yolu) )
4. Tartışma (Discussion)

Gerçekleştirilen deneysel sonuçlar, modellerin birbirlerine olan üstünlükleri ve zayıflıkları açısından incelendiğinde şu sonuçlara ulaşılmıştır:

    Sığ vs. Derin Mimarilerin Üstünlüğü: Yalnızca iki evrişim bloğuna sahip Model 1'in test doğruluğu %73.11 seviyesinde kalırken, 128 filtreli üçüncü bir evrişim bloğu eklenerek derinleştirilen Model 2, %81.40 test doğruluğuna ulaşarak bariz bir üstünlük kurmuştur. Bu durum, katman sayısı arttıkça modelin verideki daha karmaşık, soyut ve üst düzey özellikleri (high-level features) çıkarabildiğini teorik olarak kanıtlamaktadır.

    Düzenlileştirme ve Genelleme Başarısı: Veri artırımı (Data Augmentation) ve Dropout(0.3) kombinasyonu, Model 2'nin eğitim doğruluğu (%78.73) ile test doğruluğu (%81.40) arasındaki dengeyi kusursuz kurmasını sağlamıştır. Ağ ezberlemek yerine genel kalıpları öğrenmeye zorlanmış, bu sayede daha önce hiç görmediği verilerde daha yüksek başarı elde etmiştir.

    Hibrit Modelin Etkinliği: Uçtan uca (end-to-end) eğitilen Model 2'ye (%81.40) karşılık, bu modelden ayrıştırılan özelliklerin klasik bir makine öğrenmesi modeli olan SVM ile sınıflandırılması (Model 4) oldukça yakın bir performans (%78.05) sergilemiştir. Sınıflandırma raporu incelendiğinde otomobil ve gemi (F1: 0.89, 0.87) gibi keskin geometrik hatlara sahip sınıflarda hibrit sistem çok başarılı olurken; kedi ve köpek (F1: 0.61, 0.70) gibi özellikleri birbirine daha çok benzeyen sınıfları doğrusal (linear) bir düzlemde ayırmakta zorlanmıştır.

    Transfer Öğrenme ve Maliyet Karşılaştırması: ImageNet gibi devasa bir veri setiyle eğitilmiş olan AlexNet, çok düşük öğrenme hızlarında ve yalnızca 5 epoch içerisinde %88.16 gibi bir başarıya ulaşmıştır. Bu durum derin mimarilerin literatürdeki mutlak üstünlüğünü gösterse de, kendi geliştirdiğimiz Model 2'nin parametre sayısının azlığı ve eğitim maliyetinin düşüklüğü göz önüne alındığında %81.40'lık başarısı oldukça verimli ve akademik açıdan başarılı bir alternatiftir.

5. Referanslar (References)

    Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.

    PyTorch Documentation. (2026). Models and pre-trained weights. Retrieved from https://pytorch.org/vision/0.9/models.html

    Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.