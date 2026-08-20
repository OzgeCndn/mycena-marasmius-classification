# Mycena & Marasmius Deep Learning Sınıflandırma Pipeline'ı

Bu proje, Fatih Ekinci ve ekibinin yayınladığı şu makaleyi yeniden üretmek için hazırlanmıştır:

> Ekinci, F., Ugurlu, G., Ozcan, G.S., Acici, K., Asuroglu, T., Kumru, E., Guzel, M.S., Akata, I.
> **"Classification of Mycena and Marasmius Species Using Deep Learning Models: An Ecological and
> Taxonomic Approach."** *Sensors* 2025, 25, 1642. https://doi.org/10.3390/s25061642

## Sonuçlar

Pipeline, kendi indirdiğim veri setiyle (1582 görsel, 7 sınıf, %70/%30 stratified split) uçtan uca
çalıştırıldı ve 12 modelin tamamı için gerçek eğitim + değerlendirme yapıldı. Tam sonuçlar
`outputs/table5_performance.md` ve `outputs/table6_chisquare.md` dosyalarında; özet:

| Model                              | Accuracy | F1-Score | MCC   |
|:------------------------------------|---------:|---------:|------:|
| **Ensemble (MaxViT-S + ResNetV2-50)** | **0.966** | **0.966** | **0.961** |
| MaxViT-SOM (hibrit)                 |    0.949 |    0.949 | 0.941 |
| MaxViT-Small (transfer learning)    |    0.941 |    0.939 | 0.931 |
| VGG19                               |    0.928 |    0.928 | 0.916 |
| ResNetV2-50                         |    0.924 |    0.922 | 0.912 |
| MobileNetV3-Large                   |    0.918 |    0.917 | 0.905 |
| GoogleNet                           |    0.913 |    0.913 | 0.899 |
| EfficientNetV2-M                    |    0.909 |    0.909 | 0.894 |
| EfficientNet-B0                     |    0.884 |    0.884 | 0.865 |
| Custom CNN (sıfırdan)               |    0.567 |    0.559 | 0.496 |
| CNN-SOM (hibrit)                    |    0.463 |    0.466 | 0.374 |
| CNN-KAN (hibrit)                    |    0.359 |    0.339 | 0.255 |

**Öne çıkanlar:**

- En iyi sonucu, MaxViT-Small ile ResNetV2-50'nin özelliklerini birleştiren **ensemble model**
  verdi: %96.6 doğruluk, 0.998 AUC. Ki-kare anlamlılık testine (`table6_chisquare.md`) göre bu
  model; GoogleNet, MobileNetV3-Large, ResNetV2-50, EfficientNet-B0, EfficientNetV2-M ve VGG19'dan
  istatistiksel olarak anlamlı şekilde daha iyi (p < 0.05, "proposed" olarak işaretli); yalnızca
  kendi bileşeni olan MaxViT-Small'dan anlamlı farkı yok (p = 0.089), ki bu beklenen bir sonuç.
- **Transfer learning** (ImageNet ön-eğitimli ağırlıklarla başlayan 7 model) genel olarak güçlü
  performans verdi (%88-96 doğruluk), makalenin ana bulgusuyla tutarlı.
- **Sıfırdan eğitilen Custom CNN** (%56.7) ile ondan türeyen hibrit modeller **CNN-SOM** (%46.3) ve
  özellikle **CNN-KAN** (%35.9) belirgin şekilde daha düşük kaldı. Ki-kare testi bunu doğruluyor:
  CNN-SOM ve CNN-KAN, 7 temel modelin *tamamından* istatistiksel olarak anlamlı şekilde daha kötü
  (p değerleri 1e-43 ile 1e-78 arası -- yani şansla açıklanamayacak kadar büyük bir fark).
  Bunun en olası nedeni, 1582 görsellik nispeten küçük bir veri setinde sıfırdan (ImageNet
  ön-eğitimi olmadan) eğitilen bir CNN gövdesinin yeterli genel özellik öğrenememesi; ayrıca
  CNN-KAN'daki KAN katmanı makalede tam mimarisi verilmediği için burada yorumlanarak
  uygulandı (bkz. aşağıdaki "SOM ve KAN implementasyonları" notu), bu da ek bir performans
  farkına yol açmış olabilir.
- **MaxViT-SOM** (MaxViT-Small özelliklerini bir SOM sınıflandırıcıyla birleştiren hibrit),
  MaxViT-Small'ın kendisine yakın ama biraz daha iyi bir sonuç verdi (%94.9 vs %94.1) -- güçlü bir
  özellik çıkarıcı üzerine kurulan SOM'un, zayıf bir özellik çıkarıcı (Custom CNN) üzerine
  kurulandan çok daha iyi çalıştığını gösteriyor.

Bu sonuçlar, GPU'lu bir makinede tam eğitim (`python3 -m src.main --stage all`) çalıştırılarak elde
edildi; `configs/hyperparams.yaml` içindeki hiperparametre arama ve epoch sayıları makaledeki
değerlere yakın tutuldu (bkz. aşağıdaki "Kurulum" ve "Sınırlamalar" bölümleri).

## Neden bu makale seçildi

Fatih Ekinci'nin mantarlarla ilgili yaklaşık 10 derin öğrenme makalesi var, ama çoğunda dataset
"talep üzerine" paylaşılıyor (GBIF accession numaraları ile, doğrudan indirme linki yok). **Bu
makale istisna**: Data Availability Statement'ında doğrudan bir Google Drive linki var:

```
https://tinyurl.com/wp8wefy8  →  Classes.7z  (Google Drive dosya id: 1_kKt2WOXKLLqBLaGG8DkJU_hnajRHA7y)
```

Bu link WebFetch ile doğrulandı ve halka açık bir `Classes.7z` arşivine yönlendiriyor.
**Önemli**: Bu çalıştığım bulut ortamının ağ erişimi kısıtlı (yalnızca paket depolarına izin
veriliyor), bu yüzden dosyayı buradan indiremedim — sizin kendi bilgisayarınızdan indirmeniz
gerekiyor. Aşağıda adımlar var.

## Veri setini indirme

1. Tarayıcıda şu linki açın: https://drive.google.com/file/d/1_kKt2WOXKLLqBLaGG8DkJU_hnajRHA7y/view
2. "Classes.7z" dosyasını indirin (veya kendi ortamınızda: `pip install gdown` sonra
   `gdown 1_kKt2WOXKLLqBLaGG8DkJU_hnajRHA7y -O Classes.7z`)
3. Arşivi bu projenin `data/` klasörüne çıkarın, öyle ki yapı şöyle olsun:

```
data/
  Classes/
    Marasmius_oreades/      (222 görsel)
    Marasmius_rotula/       (228 görsel)
    Mycena_crocata/         (229 görsel)
    Mycena_epipterygia/     (243 görsel)
    Mycena_pura/            (220 görsel)
    Mycena_rosea/           (227 görsel)
    Mycena_seynii/          (213 görsel)
```

   (Toplam 1582 görsel — makaledeki Table 4 ile eşleşmeli. Klasör isimleri arşivde farklıysa
   `configs/hyperparams.yaml` içindeki `class_names` listesini ona göre güncelleyin; `src/data.py`
   klasör isimlerini olduğu gibi kullanır.)

## Kurulum

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

GPU'nuz varsa PyTorch otomatik kullanır (`torch.cuda.is_available()`); yoksa CPU'da çalışır ama
**8 CNN/transformer modeli + 3 hibrit model + 1 ensemble = 12 model** eğitmek CPU'da muhtemelen
saatler sürer (makalede modeller 10-33 epoch arası eğitiliyor).

## Çalıştırma

Proje kökünden, modül olarak çalıştırın (göreli import'lar nedeniyle `-m` gerekli):

```bash
# 1) Tüm modelleri eğit (hyperparametre random search + belirtilen epoch sayılarıyla)
python3 -m src.main --data-dir data/Classes --output-dir outputs --stage train

# 2) Test seti üzerinde değerlendir, Table 5 ve Table 6'yı üret
python3 -m src.main --data-dir data/Classes --output-dir outputs --stage evaluate

# Veya ikisini birden:
python3 -m src.main --data-dir data/Classes --output-dir outputs --stage all
```

Tek bir modeli eğitmek/denemek için:

```bash
python3 -m src.main --data-dir data/Classes --output-dir outputs --stage all --models maxvit_s
```

## Ne üretiyor

- `outputs/checkpoints/<model>.pt` — her modelin en iyi ağırlıkları (bu repoya dahil değil, bkz.
  `.gitignore` — 10 model ağırlığı toplam 1.5GB'ın üzerinde, GitHub'ın dosya boyutu limitini aşıyor)
- `outputs/table5_performance.csv` / `.md` — makaledeki **Table 5**: her model için
  Accuracy, Precision, Recall, F1, Specificity, MCC, AUC (OvR), makro-ortalama (bu repoda mevcut,
  yukarıdaki "Sonuçlar" bölümündeki tablo buradan alınmıştır)
- `outputs/table6_chisquare.csv` / `.md` — makaledeki **Table 6**: önerilen modeller
  (CNN-SOM, CNN-KAN, MaxViT-SOM, Ensemble) ile temel modeller arasında ki-kare anlamlılık testi
  (p-değerleri; bu repoda mevcut)
- `outputs/confusion_matrices/` — her model için karışıklık matrisi (PNG)

## Modeller (makaledeki Bölüm 2.3-2.4 ile birebir)

| Model | Kaynak | Epoch |
|---|---|---|
| Custom CNN | `src/models/custom_cnn.py` (4 conv blok, 32→64→128→256 filtre) | 33 (early stopping) |
| GoogleNet | torchvision, ImageNet önceden eğitilmiş | 10 |
| MobileNetV3-Large | torchvision, ImageNet önceden eğitilmiş | 10 |
| VGG19 | torchvision, ImageNet önceden eğitilmiş | 20 |
| ResNetV2-50 | timm (`resnetv2_50`) | 20 |
| EfficientNet-B0 | torchvision, ImageNet önceden eğitilmiş | 20 |
| EfficientNetV2-M | timm (`efficientnetv2_m`) | 20 |
| MaxViT-Small | timm (`maxvit_small_tf_224`) | 20 |
| CNN-SOM | Custom CNN özellik çıkarıcı + 20×20 SOM (BMU sınıflandırma) | SOM: 5000 iterasyon |
| CNN-KAN | Custom CNN + Kolmogorov-Arnold katmanı | Custom CNN ile aynı |
| MaxViT-SOM | MaxViT-S özellik çıkarıcı (768-d) + 10×10 SOM | SOM: 1000 iterasyon |
| Ensemble (MaxViT-S + ResNetV2-50) | Paralel özellik çıkarma → concat (2816-d) → FC(512)→FC(7) | 20 |

**Not — SOM ve KAN implementasyonları:** Makale "MiniSom" kütüphanesini kullandığını belirtiyor,
ama bu bulut ortamında `minisom` paketinin derlenmesi başarısız oldu (setuptools uyumsuzluğu), bu
yüzden `src/models/som.py` içinde NumPy tabanlı, MiniSom ile aynı API'ye ve algoritmaya (Gaussian
komşuluk fonksiyonu, BMU) sahip minimal bir SOM sınıfı yazıldı. Kendi makinenizde `pip install
minisom` çalışıyorsa `src/models/som.py` içindeki `USE_MINISOM = True` yapıp orijinal kütüphaneyi
kullanabilirsiniz. KAN katmanı için makalede "sine-based transformation with learnable activation
functions" dışında mimari detay verilmemiş; `src/models/kan.py` içindeki katman bunun makul bir
yorumudur — yazarların orijinal kodu yayımlanmadığı için birebir aynı olduğu garanti edilemez.

## Değerlendirme metrikleri (Bölüm 2.6, Formül 12-18)

`src/metrics.py` şunları hesaplar (7 sınıf için makro ortalama):
Accuracy, Precision, Recall/Sensitivity, Specificity, F1, MCC (multi-class formülü), AUC (One-vs-Rest).
Table 6 için karışıklık matrislerinden ki-kare testi (`scipy.stats.chi2_contingency`) uygulanır.

## Doğrulama / sağlık kontrolü

Gerçek veri olmadan da pipeline'ın uçtan uca çalıştığını doğrulamak için sentetik (rastgele)
görsellerle bir smoke test dahil edildi:

```bash
python3 tests/smoke_test.py
```

Bu, her model mimarisinin kurulup forward-pass yapabildiğini, veri yükleyicinin doğru
train/test bölmesi ürettiğini ve tablo üretim kodunun doğru şekil/format çıktısı verdiğini birkaç
saniyede doğrular (gerçek eğitim yapmaz).

## Sınırlamalar / dürüst notlar

- Görsellerin tam olarak makaledeki 1582 görselle birebir aynı olduğunu doğrulamak sizin elinizde
  (indirdiğiniz `Classes.7z` içeriğini `data/Classes/*/*.jpg` sayımıyla Table 4'teki sayılarla
  karşılaştırın — `tests/check_dataset.py` bunu otomatik yapar).
- Makalede "random search" ile hiperparametre optimizasyonu belirtiliyor ama arama uzayındaki kaç
  deneme yapıldığı / hangi tohum (seed) kullanıldığı belirtilmemiş; `configs/hyperparams.yaml`
  içinde arama uzayı makaledeki gibi tanımlı, deneme sayısı makul bir varsayılan (`n_trials: 10`)
  olarak ayarlandı — isterseniz artırabilirsiniz.
- Ki-kare tablosundaki ok işaretleri (↑/←) makalede görsel/yorumsal notasyon; kod sadece p-değerini
  üretir, anlamlılık yönünü (`outputs/table6_chisquare.csv` içindeki `favored_model` sütunu) ayrıca
  hesaplar.
- Yukarıdaki "Sonuçlar" bölümündeki sayılar gerçek bir eğitim koşusundan elde edildi, ancak
  `n_trials`, batch size ve random seed gibi seçimler makaledeki tam değerlerle birebir eşleşmeyebilir;
  bu yüzden makaledeki sayılarla küçük farklar olması beklenir.
