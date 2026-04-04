# Deteksi Warna

## Gambaran Umum

Sistem drone-seeker mendeteksi target berwarna merah-muda terang (hot-pink) menggunakan pipeline computer vision bertingkat di `seeker.py`. Deteksi berbasis kalibrasi: histogram hue yang disimpan oleh `calibrate_color.py` mendefinisikan model warna target. Tanpa file kalibrasi, sistem menggunakan rentang HSV hot-pink yang dikodekan langsung sebagai fallback.

---

## 1. Kalibrasi (`calibrate_color.py`)

Sebelum terbang, jalankan `calibrate_color.py` untuk membangun model warna target sesuai kondisi pencahayaan nyata.

### Alur Kerja

1. Arahkan kamera ke target hot-pink.
2. Alat mendeteksi blob menggunakan histogram kepercayaan yang tersimpan atau fallback HSV yang dikodekan.
3. Histogram hue 180-bin diakumulasi dari region blob yang terdeteksi.
4. Tekan **S** untuk menyimpan histogram yang telah dinormalisasi ke `color_histogram.txt`.
5. Tekan **R** untuk mereset ke fallback HSV.

![Alur kalibrasi warna](chart_01_calibration.png)

### Format file yang disimpan

ASCII biasa, 180 baris — satu bobot floating-point yang dinormalisasi per bin hue (OpenCV hue 0–179).

---

## 2. Fitting Model Warna (`seeker.py`)

Saat startup, `Seeker` memuat histogram kalibrasi dan memasang model statistik pada histogram tersebut.

### 2.1 Fitting Gaussian Sirkular (`_fit_gaussian`)

Hot pink melintasi batas hue 0/179, sehingga rata-rata aritmatika biasa tidak bermakna. Sebagai gantinya:

1. Ubah setiap indeks bin menjadi sudut pada lingkaran satuan:
   `θ_i = i × 2π / 180`
2. Hitung jumlah vektor satuan berbobot dari semua 180 bin:
   `(cx, cy) = Σ p_i (cos θ_i, sin θ_i)`
3. Rata-rata sirkular: `μ = atan2(cy, cx)` dipetakan kembali ke [0, 179].
4. Standar deviasi sirkular: bungkus jarak bin ke [−90, 90] relatif terhadap μ, lalu hitung varians berbobot → `σ`.

```python
def _fit_gaussian(hist: np.ndarray) -> tuple[float, float]:
    bins = np.arange(180, dtype=np.float32)
    h    = hist.flatten().astype(np.float32)
    total = h.sum()
    if total == 0:
        return 90.0, 30.0

    prob = h / total

    # Rata-rata sirkular via rata-rata vektor satuan
    angles = bins * (2.0 * np.pi / 180.0)
    cx = float(np.sum(np.cos(angles) * prob))
    cy = float(np.sum(np.sin(angles) * prob))
    mean_rad = np.arctan2(cy, cx)
    if mean_rad < 0:
        mean_rad += 2.0 * np.pi
    mean_hue = float(mean_rad * 180.0 / (2.0 * np.pi))

    # Std sirkular: bungkus bin ke [-90, 90] relatif terhadap mean, lalu hitung varians
    diff = bins - mean_hue
    diff = ((diff + 90.0) % 180.0) - 90.0   # bungkus ke [-90, 90]
    var  = float(np.sum(diff ** 2 * prob))
    return mean_hue, float(np.sqrt(max(var, 1.0)))
```

### 2.2 Histogram Kepercayaan (`_confidence_hist`)

Nolkan setiap bin yang jarak sirkularnya dari μ melebihi **3.0 σ**:

```python
def _confidence_hist(hist: np.ndarray, mean: float, std: float) -> np.ndarray:
    bins = np.arange(180, dtype=np.float32)
    diff = np.abs(bins - mean)
    diff = np.minimum(diff, 180.0 - diff)          # pembungkusan sirkular
    conf = hist.flatten().copy().astype(np.float32)
    conf[diff >= _GAUSS_SIGMA * std] = 0.0
    return conf.reshape(hist.shape)
```

Hanya nilai hue yang benar-benar milik target (dalam 3.0 standar deviasi) yang berkontribusi pada deteksi. Histogram ini digunakan untuk semua back-projection di tahap selanjutnya.

---

## 3. Pipeline Deteksi (`_detection_mask`)

Setiap frame menjalankan langkah-langkah berikut:

### Langkah 0 — Path Cepat Saat Terkunci

Ketika tracker sudah mengunci target (`locked = True` — `_detect_count >= 3` dan `_track_win` aktif), pipeline penuh dilewati. Hanya **Metode 3 (inRange)** yang dijalankan, diikuti oleh satu operasi `MORPH_CLOSE` 3×3. Ini sekitar 3–4× lebih cepat per frame dan cukup karena target telah dikonfirmasi.

```python
if locked:
    mask = self._mask_inrange(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kern3)
    return mask, 1
```

### Langkah 1 — Blur Hue (jalur akuisisi)

Gaussian blur 5×5 pada kanal H menekan noise hue per-piksel dari artefak JPEG, demosaicing, dan sorotan spekuler:

```python
h_blur = cv2.GaussianBlur(hsv[:, :, 0], (5, 5), 0)
```

Blur ini hanya digunakan oleh Metode 1 dan 2. Metode 3 beroperasi langsung pada HSV mentah.

### Langkah 2 — Tiga Mask Independen

Tiga metode masing-masing menghasilkan mask biner secara independen. Piksel **diterima bila minimal 2 dari 3 metode setuju** (voting mayoritas).

#### Metode 1 — Back-projection Gaussian (`_mask_gaussian`)

Memproyeksikan histogram kepercayaan kembali ke kanal hue yang diblur. Bin yang lebih dekat ke μ memiliki bobot lebih tinggi, sehingga histogram kepercayaan mengkodekan distribusi yang telah dipelajari, bukan hanya gerbang.  Implementasi menukar channel H sementara di buffer yang sudah dialokasikan (tanpa `copy()`):

```python
def _mask_gaussian(self, hsv: np.ndarray, h_blur: np.ndarray) -> np.ndarray:
    np.copyto(self._h_buf, hsv[:, :, 0])      # simpan H asli
    hsv[:, :, 0] = h_blur                      # ganti sementara dengan H yang diblur
    bp = cv2.calcBackProject([hsv], [0], self._conf_hist, [0, 180], 1)
    hsv[:, :, 0] = self._h_buf                 # pulihkan H asli
    sv_ok = self._apply_inrange_band(hsv, "outer")   # gerbang S/V via band precomputed
    return cv2.bitwise_and(cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY)[1], sv_ok)
```

Gerbang S/V menggunakan `"outer"` band (μ ± 3.0σ pada S minimum 40, V minimum 40) yang sudah dihitung sekali saat inisialisasi.

#### Metode 2 — Threshold Hue Adaptif (`_mask_adaptive`)

Menemukan piksel yang hue-nya konsisten secara lokal, menangani iluminasi tidak merata di seluruh frame:

1. Terapkan `adaptiveThreshold` (blockSize=11, C=3) langsung pada `h_blur` — setiap piksel dibandingkan dengan rata-rata berbobot Gaussian dari lingkungan 11×11-nya. (`blockSize` diturunkan dari 21 menjadi 11 untuk separuh biaya komputasi.)
2. Gate dengan LUT yang telah dikalkulasi: `_hue_gate_lut[h_blur]` — True di mana bin hue berada dalam ±3.0σ dari μ.

```python
def _mask_adaptive(self, hsv: np.ndarray, h_blur: np.ndarray) -> np.ndarray:
    adapt    = cv2.adaptiveThreshold(
        h_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=3,
    )
    hue_gate = self._hue_gate_lut[h_blur]   # LUT precomputed: 255 dalam ±σ*std, 0 lainnya
    return cv2.bitwise_and(adapt, hue_gate)
```

Robust terhadap scene di mana satu sisi target lebih terang dari sisi lainnya.

#### Metode 3 — Dual inRange (`_mask_inrange`)

Membangun band HSV dari bounds yang sudah dikalkulasi sebelumnya (`_precomp_bands`). Rentang dibagi menjadi **core** (μ ± 1σ) dan **outer** (μ ± 3.0σ). Piksel outer-only mendapat bobot setengah (128) untuk menyampaikan keyakinan gradual — core dapat penuh (255):

```python
def _mask_inrange(self, hsv: np.ndarray) -> np.ndarray:
    core  = self._apply_inrange_band(hsv, "core")
    outer = self._apply_inrange_band(hsv, "outer")
    # piksel outer-saja mendapat bobot setengah; core tetap 255
    cv2.subtract(outer, core, dst=outer)       # hanya sisakan outer-tapi-bukan-core
    outer = cv2.LUT(outer, self._outer_lut)    # 255 → 128, 0 → 0
    return cv2.bitwise_or(core, outer)
```

Wrap hue (mis. hot pink melewati batas 0/179) ditangani otomatis oleh `_apply_inrange_band` yang mengeluarkan dua panggilan `cv2.inRange` dan melakukan OR jika diperlukan.

### Langkah 3 — Voting Mayoritas

```python
votes = (m1 > 0).view(np.uint8)
votes += (m2 > 0).view(np.uint8)
votes += (m3 > 0).view(np.uint8)
_, mask = cv2.threshold(votes, 1, 255, cv2.THRESH_BINARY)  # ≥ 2 dari 3
```

### Langkah 4 — Pembersihan Morfologis

Satu operasi `MORPH_CLOSE` dengan kernel 3×3 menggantikan urutan `OPEN + DILATE` sebelumnya. CLOSE lebih hemat (satu operasi vs dua) dan mengisi celah kecil dalam blob sekaligus menekan noise terisolasi:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # tutup celah kecil dalam blob
```

### Fallback (tanpa file kalibrasi)

Jika tidak ada file histogram, deteksi menggunakan satu band hot-pink yang dikodekan langsung:

| H | S | V | Keterangan |
|---|---|---|---|
| 130–173 | 40–233 | 80–233 | Hot pink / magenta |

---

## 4. Seleksi Blob (`_nearest_blob_rect`)

`cv2.findContours` mengekstrak semua kontur eksternal dari mask. Setiap kandidat diuji dengan **empat filter bentuk** sebelum dinilai:

| Konstanta | Nilai | Peran |
|---|---|---|
| `_MIN_BLOB_AREA` | 20 px² | Hapus noise terlalu kecil |
| `_MIN_DIM` | 4 px | Lebar DAN tinggi minimal (menolak garis tipis) |
| `_MAX_ASPECT` | 6.0 | Rasio sisi panjang/pendek maksimum (menolak sliver) |
| `_MIN_EXTENT` | 0.45 | `area_kontur / area_bbox` minimum (menolak bentuk L/U) |
| `_MIN_SOLIDITY` | 0.60 | `area_kontur / area_convex_hull` minimum (menolak bentuk cekung) |

Kandidat yang lolos filter dinilai dengan:

```
skor = solidity × extent × area
```

Blob yang paling **kompak, terisi, dan besar** menang. Parameter opsional `prefer_pt=(px, py)` membagi skor dengan `(1 + jarak/100)` sehingga blob yang lebih dekat ke titik referensi menang pada nilai yang seri:

```python
_MIN_BLOB_AREA = 20   # px²
_MIN_EXTENT    = 0.45
_MIN_SOLIDITY  = 0.60
_MIN_DIM       = 4
_MAX_ASPECT    = 6.0

def _nearest_blob_rect(mask, frame_shape=None, box_filter=True, prefer_pt=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score, best_rect = -1.0, None
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if box_filter:
            if w < _MIN_DIM or h < _MIN_DIM:
                continue
            if max(w, h) / min(w, h) > _MAX_ASPECT:
                continue
            if area / (w * h) < _MIN_EXTENT:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / hull_area if hull_area > 0 else 0.0
            if solidity < _MIN_SOLIDITY:
                continue
            score = solidity * (area / (w * h)) * area
        else:
            score = area
        if prefer_pt is not None:
            dist = ((x + w*0.5 - prefer_pt[0])**2 + (y + h*0.5 - prefer_pt[1])**2)**0.5
            score /= (1.0 + dist / 100.0)
        if score > best_score:
            best_score, best_rect = score, (x, y, w, h)
    return best_rect
```

Mengembalikan bounding rectangle berorientasi sumbu `(x, y, w, h)`, atau `None` jika tidak ada blob valid yang ditemukan. Saat `box_filter=False` (jalur re-akuisisi CamShift), hanya ambang area minimum yang diterapkan.

---

## Diagram Ringkasan

![Pipeline deteksi warna](chart_01_detection.png)
