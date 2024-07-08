# Sıfırdan Sinir Ağları

Bu depo, Python ve NumPy kullanarak sıfırdan inşa edilen çeşitli sinir ağı uygulamalarını içermektedir. Her bir uygulama, farklı sinir ağı türlerini, mimarilerini ve bu ağların gerçek dünya problemlerini nasıl çözdüğünü göstermektedir.

## İçindekiler

1. [Giriş](#giriş)
2. [Başlarken](#başlarken)
3. [İkili Sınıflandırma](#ikili-sınıflandırma)
4. [Katkıda Bulunma](#katkıda-bulunma)
5. [Lisans](#lisans)

## Giriş

Bu depoda, sinir ağlarının iç işleyişini keşfederek sıfırdan uygulamalarını inşa ediyoruz. Amacımız, bu güçlü modelleri yöneten temel ilkeleri ve matematiği derinlemesine anlamaktır. Sinir ağlarını sıfırdan inşa ederek, nasıl öğrendiklerini, optimize ettiklerini ve tahminlerde bulunduklarını daha iyi anlıyoruz.

## Başlarken

Bu projeye başlamak için depoyu klonlamanız ve gerekli bağımlılıkları yüklemeniz gerekmektedir. Aşağıdaki adımları izleyin:

### Gereksinimler

- Python 3.x
- NumPy
- Pandas

### Kurulum

1. Depoyu klonlayın:

   ```bash
   git clone https://github.com/beratcmn/nnfs.git
   cd nnfs
   ```

2. Bağımlılıkları yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

## İkili Sınıflandırma

Bu bölümde, ikili sınıflandırma için kullanılan basit ama öğretici bir sinir ağı örneğine odaklanacağız. Bir veri seti ile iki özellik ve bir ikili etiket kullanarak, veri noktalarını iki kategoriden birine sınıflandırabilen bir sinir ağı eğiteceğiz.

### Genel Bakış

İkili sınıflandırma örneği aşağıdaki ana kavramları göstermektedir:

- Veri yükleme ve ön işleme
- Sinir ağı mimarisi
- İleri yayılım
- Geri yayılım ve gradyan inişi
- Eğitim ve değerlendirme

### Kod Açıklaması

#### Veri Yükleme ve Ön İşleme

Veri setini yükleyerek ve özellikleri ve etiketleri çıkararak başlıyoruz. Veri setinin CSV formatında olduğu varsayılmaktadır.

```python
import pandas as pd
import numpy as np

# Yeniden üretilebilirlik için rastgele tohum belirleyin
np.random.seed(36)

# CSV'den verileri yükleyin
data = pd.read_csv("./data/binary_500.csv")
X = data[["Feature1", "Feature2"]].values
Y = data["Label"].values

# Etiketleri 2D bir diziye dönüştürün
Y = Y.reshape(-1, 1)
```

#### Ağ Mimarisi

![Sinir Ağı Mimarisi](https://raw.githubusercontent.com/beratcmn/nnfs/main/assets/binary.png)

Bir gizli katmana sahip basit bir sinir ağı tanımlıyoruz. Ağın mimarisi şu şekildedir:

- Giriş katmanı: 2 nöron (her özellik için bir)
- Gizli katman: 3 nöron
- Çıkış katmanı: 1 nöron (ikili çıkış)

```python
# Ağ mimarisi
input_size = 2
hidden_size = 3
output_size = 1

# Ağırlıklar ve önyargıları başlatma
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
```

#### İleri Yayılım

İleri yayılım fonksiyonu, sigmoid aktivasyon fonksiyonunu kullanarak her katmandaki nöronların aktivasyonlarını hesaplar.

```python
# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X):
    # Girişten gizli katmana
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # Gizliden çıkış katmanına
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A1, A2
```

#### Geri Yayılım ve Gradyan İnişi

Geri yayılım fonksiyonu, kayıp fonksiyonunun ağırlıklar ve önyargılar ile ilgili gradyanlarını hesaplar ve bunları gradyan inişi kullanarak günceller.

```python
def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(X, Y, A1, A2, learning_rate=0.1):
    global W1, b1, W2, b2

    # Hata hesapla
    error = Y - A2
    dA2 = error * sigmoid_derivative(A2)

    # W2 ve b2 için gradyanları hesapla
    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    # W1 ve b1 için gradyanları hesapla
    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    # Ağırlıkları ve önyargıları güncelle
    W1 += learning_rate * dW1
    b1 += learning_rate * db1
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
```

#### Sinir Ağını Eğitme

`train` fonksiyonu, ileri yayılım ve geri yayılım işlemlerini yineleyerek kayıp fonksiyonunu minimize eder ve ağın tahminlerini iyileştirir.

```python
def train(X, Y, epochs=1000, learning_rate=0.1):
    for epoch in range(epochs):
        A1, A2 = forward_propagation(X)
        backpropagation(X, Y, A1, A2, learning_rate)

        if epoch % 100 == 0:
            # Kayıp fonksiyonunu hesapla (ortalama kare hata)
            loss = np.mean((Y - A2) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

#### Tahmin Yapma

`predict` fonksiyonu, eğitilmiş ağı kullanarak yeni veriler üzerinde tahminler yapar.

```python
def predict(X):
    _, A2 = forward_propagation(X)
    return (A2 > 0.5).astype(int)

# Eğitim verileri üzerinde tahmin yap
predictions = predict(X)

# Doğruluğu hesapla
accuracy = np.mean(predictions == Y)
print(f"Accuracy: {accuracy:.4f}")
```

### Özet

Bu örnek, ikili sınıflandırma görevleri için sıfırdan bir sinir ağının nasıl uygulanabileceğini açık ve net bir şekilde göstermektedir. Her bileşenin ve ağdaki rolünün anlaşılmasıyla, sinir ağı ilkeleri ve pratik uygulamaları hakkında sağlam bir temel kazanacaksınız.

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen bir Pull Request gönderin.
