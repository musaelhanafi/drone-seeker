# Tinjauan Pustaka

Tinjauan ini mengkaji literatur dasar dan terapan yang mendukung pipeline deteksi warna, visual tracking, dan aktuasi MAVLink yang diimplementasikan dalam drone-seeker. Karya-karya dikelompokkan berdasarkan area topik dan dibahas dalam konteks keputusan desain yang dibuat dalam sistem ini.

---

## 1. Pemilihan Ruang Warna — HSV

Pemilihan ruang warna HSV (Hue–Saturation–Value) sebagai dasar deteksi berasal langsung dari kerangka formal yang diperkenalkan oleh Smith [1]. Dengan memisahkan informasi kromatik (Hue) dari luminansi (Value), HSV memisahkan identitas warna suatu objek dari iluminasi lingkungan yang jatuh padanya. Hal ini sangat penting dalam aplikasi drone outdoor di mana intensitas dan sudut sinar matahari berubah terus-menerus selama penerbangan. Sifat geometris dan perseptual HSV dibahas secara komprehensif oleh Gonzalez dan Woods [2], yang pembahasannya tentang segmentasi warna tetap menjadi referensi teknik standar. Dalam drone-seeker, ketiga metode deteksi beroperasi dalam HSV: metode back-projection Gaussian membangun histogram kepercayaan pada kanal hue; metode adaptif menormalisasi dan mengenakan threshold pada kanal hue secara lokal; dan metode dual inRange menerapkan batas per-kanal langsung dalam ruang HSV.

---

## 2. Back-Projection Histogram

Detektor back-projection Gaussian dalam drone-seeker merupakan penerapan langsung dari kerangka color indexing yang diperkenalkan oleh Swain dan Ballard [3]. Makalah seminal mereka tahun 1991 menunjukkan bahwa histogram warna suatu objek referensi dapat digunakan untuk menghitung, untuk setiap piksel gambar kueri, probabilitas bahwa piksel tersebut milik objek tersebut — sebuah proses yang mereka sebut back-projection. Peta probabilitas yang dihasilkan kemudian di-threshold untuk mendapatkan mask deteksi. Dalam drone-seeker, `cv2.calcBackProject` dipanggil dengan histogram kepercayaan Gaussian yang telah dibuat (`_confidence_hist`) yang hanya memberikan probabilitas tinggi pada bin hue dalam ±2.5σ dari hue mean yang dikalibrasi, menekan semua hue lain ke nol. Batasan Saturation dan Value diterapkan secara terpisah untuk menolak region akromatik dan gelap.

---

## 3. Statistik Sirkular untuk Pemodelan Hue

Hue adalah besaran sirkular (periodik): hue dari objek "hot pink" mungkin terpusat di sekitar H = 165 dalam skala 0–179 OpenCV, tetapi ketika pink melintasi batas (misalnya H = 175 membungkus ke H = 0), rata-rata Euclidean dan standar deviasi standar memberikan hasil yang salah. Perlakuan yang benar diberikan oleh Fisher [4] dan Mardia dan Jupp [5], yang buku teks mereka menetapkan teori statistik sirkular. Rata-rata distribusi sirkular dihitung dengan memetakan setiap sudut ke vektor satuan, merata-ratakan vektor, dan mengambil argumen resultan — metode yang menangani pembungkusan secara alami. Standar deviasi sirkular (atau deviasi sudut) diturunkan dari besaran resultan tersebut. Fungsi `_fit_gaussian` drone-seeker mengimplementasikan ini secara langsung untuk kalibrasi hue.

Kebutuhan teoritis untuk aritmetika sirkular pada Hue lebih lanjut ditunjukkan oleh Hanbury dan Serra [6], yang menunjukkan bahwa operator morfologis dan metrik jarak yang diterapkan secara naif pada kanal Hue — dengan memperlakukannya sebagai variabel linear — menghasilkan hasil yang tidak konsisten secara topologis. Karya mereka memotivasi penggunaan jarak sirkular dalam konstruksi `_confidence_hist` dan dalam pembungkusan sudut yang diterapkan selama thresholding adaptif.

---

## 4. Thresholding Adaptif

Metode deteksi adaptif dalam drone-seeker dimotivasi oleh keterbatasan thresholding global di bawah iluminasi yang tidak merata. Sauvola dan Pietikäinen [7] menetapkan prinsip binarisasi adaptif lokal: threshold pada setiap piksel diturunkan dari rata-rata dan standar deviasi lokal dalam lingkungan spasial, membuat keputusan tidak berubah terhadap gradien spasial lambat dalam iluminasi. Bradley dan Roth [8] merumuskan ulang pendekatan ini menggunakan integral image, mencapai komputasi yang hampir konstan terlepas dari ukuran lingkungan — properti kunci untuk pemrosesan video real-time. Dalam `_mask_adaptive` drone-seeker, kanal Hue pertama dinormalisasi ke tingkat iluminasi lokal, kemudian `cv2.adaptiveThreshold` (yang menggunakan threshold rata-rata lokal setara dengan Bradley–Roth) diterapkan dengan ukuran blok 21 piksel. Hasilnya di-mask terhadap gate hue kasar untuk mempertahankan hanya region yang plausibel secara hue.

---

## 5. Tracking CamShift dan MeanShift

### 5.1 Mean Shift

Mean Shift adalah algoritma pencarian modus non-parametrik yang menjadi fondasi CamShift. Pertama kali diperkenalkan oleh Fukunaga dan Hostetler [21] sebagai metode estimasi gradien fungsi densitas probabilitas, algoritma ini secara iteratif menggeser titik estimasi menuju region densitas tertinggi dalam ruang fitur. Comaniciu dan Meer [22] kemudian memperluas Mean Shift menjadi kerangka analisis ruang fitur yang kuat — termasuk aplikasi untuk segmentasi citra dan tracking objek — dan membuktikan sifat konvergensinya pada distribusi kernel yang umum.

Dalam konteks visual tracking, Mean Shift mempertahankan ukuran window tetap dan hanya memperbarui posisi ke modus lokal dari back-projection warna. Hasilnya adalah `cv2.Rect` axis-aligned tanpa informasi orientasi, berbeda dengan CamShift yang menghasilkan `cv2.RotatedRect`. Dalam drone-seeker, opsi `--tracker meanshift` mengaktifkan varian ini: pipeline identik dengan CamShift (back-projection kepercayaan Gaussian, pre-translate Kalman, gate, GaussianBlur) tetapi pemanggilan algoritma akhir diganti dengan `cv2.meanShift()`. Ini cocok untuk target yang tidak memerlukan estimasi orientasi dan memberikan beban komputasi yang sedikit lebih rendah.

### 5.2 CamShift

CamShift (Continuously Adaptive Mean Shift) diperkenalkan oleh Bradski [9, 10] sebagai perluasan algoritma Mean Shift ke sekuens video. CamShift tidak hanya memperbarui posisi window seperti Mean Shift, tetapi juga mengadaptasi ukuran dan orientasi window pada setiap frame agar sesuai dengan skala dan postur objek yang dilacak saat ini. Fondasi matematika dari pendekatan ini, menggunakan estimasi densitas kernel dengan histogram warna sebagai model probabilitas dasar, diformalisasi secara ketat oleh Comaniciu, Ramesh, dan Meer [11], yang tracker berbasis kernel memberikan landasan teoritis untuk implementasi CamShift praktis yang tersedia sebagai `cv2.CamShift` dalam OpenCV.

Dalam drone-seeker, CamShift diinisialisasi dengan bounding rectangle yang dikembalikan oleh langkah deteksi terakhir yang berhasil. Tracker berjalan pada peta back-projection kepercayaan Gaussian yang sama yang digunakan untuk deteksi, memastikan pemodelan warna yang konsisten antara dua tahap. Jika window tracker menyusut di bawah area minimum atau peta kepercayaan kehilangan sinyal, sistem kembali ke mode deteksi.

---

## 5a. Filter Kalman untuk Prediksi Posisi

Filter Kalman adalah estimator rekursif yang optimal dalam artian Mean Square Error untuk sistem linear dengan noise Gaussian. Diperkenalkan oleh Kalman [19] pada 1960, filter ini mempropagasi distribusi posterior state (posisi dan kecepatan) secara rekursif melalui dua tahap: *predict* — memajukan estimasi sesuai model dinamika — dan *update* — mengoreksi estimasi menggunakan pengukuran baru berbobot invers kovarians. Welch dan Bishop [20] menyediakan derivasi yang dapat diakses dalam konteks computer vision dan robotika.

Dalam drone-seeker, filter Kalman 1D diterapkan secara terpisah pada sumbu x dan y centroid yang dilacak. State vektor masing-masing adalah `[posisi, kecepatan]`, dengan parameter proses dan pengukuran:

| Parameter | Nilai | Peran |
|---|---|---|
| `_KF_Q_POS` | 2.0 px²/s | Noise proses — posisi |
| `_KF_Q_VEL` | 80.0 px²/s³ | Noise proses — kecepatan |
| `_KF_R` | 30.0 px² | Noise pengukuran |

Ketika CamShift atau tracker penampilan kehilangan target (`camshift_bad = True` atau miss_count meningkat), filter Kalman melanjutkan dengan langkah *predict-only* — memproyeksikan posisi berdasarkan kecepatan yang diestimasi terakhir — selama hingga `_KF_MISS_MAX = 5` frame sebelum hard reset. Ini memungkinkan sistem mempertahankan lock sementara selama oklusi singkat atau saturasi cahaya tanpa memicu perubahan mode yang tidak perlu.

---

## 5b. Tracker Berbasis Penampilan — MIL

Selain tracker berbasis histogram (CamShift/MeanShift), drone-seeker mendukung tracker penampilan MIL yang diaktifkan melalui `--tracker mil`. Setelah target terkunci tiga frame berturut-turut, tracker MIL diinisialisasi dengan bounding box deteksi dan berjalan setiap frame setelahnya. Karena tracker penampilan tidak secara inheren mengetahui warna target, drone-seeker menambahkan lapisan validasi warna: mask deteksi dijalankan pada ROI yang dikembalikan tracker setiap frame; jika `countNonZero(roi_mask) < _MIN_BLOB_AREA` selama `_KF_MISS_MAX` frame berurutan, tracker direset dan sistem kembali ke mode deteksi warna.

### MIL (Multiple Instance Learning)

MIL diperkenalkan oleh Babenko et al. [23] sebagai kerangka kerja generalisasi dari klasifikasi online konvensional yang diterapkan pada visual object tracking. Pendekatan konvensional memberi label positif tunggal pada patch yang tepat di posisi target — namun dalam tracking video, posisi yang diestimasi selalu memiliki ketidakpastian inheren akibat noise pengukuran dan drift gradual. Akibatnya, patch berlabel positif seringkali tidak benar-benar berada pada objek target, yang secara bertahap meracuni classifier online dan menyebabkan drift tracking.

MIL mengatasinya dengan merumuskan masalah dalam kerangka Multiple Instance Learning (Dietterich et al. [29]): alih-alih satu patch positif, setiap update training menggunakan sebuah "bag" yang berisi beberapa patch kandidat yang disampling dari lingkungan sekitar posisi target yang diestimasi. Aturan MIL menyatakan bahwa sebuah bag positif jika *setidaknya satu* instance di dalamnya adalah positif. Kondisi lemah ini cukup untuk memastikan bahwa — meskipun posisi estimasi agak meleset — patch yang benar-benar berada pada target tetap memperoleh label positif secara implisit.

Secara teknis, Babenko et al. menggunakan classifier berbasis **MILBoost**: sebuah prosedur boosting yang meminimalkan likelihood negatif di bawah model generatif MIL. Pada setiap frame, representasi fitur Haar-like diekstraksi dari patch dalam bag positif (dekat target) dan bag negatif (jauh dari target), dan boostingclassifier diperbarui secara online. Posisi target baru diestimasi sebagai patch dengan respons klasifier tertinggi dalam jendela pencarian.

Dalam implementasi OpenCV (`cv2.TrackerMIL_create()` / `cv2.legacy.TrackerMIL_create()`), classifier MILBoost ini dijalankan sepenuhnya pada CPU tanpa dependensi model eksternal. Properti kunci yang relevan untuk drone-seeker:

- **Tidak memerlukan file model** — tidak ada bobot jaringan yang perlu diunduh; tracker diinisialisasi ulang dari awal setiap kali target dikunci kembali.
- **Kompatibilitas OpenCV** — tersedia di OpenCV inti (tidak memerlukan `opencv-contrib`), sehingga dapat dikompilasi dan dijalankan di companion computer ARM tanpa dependensi tambahan.
- **Keterbatasan deteksi kehilangan** — MIL tidak memiliki mekanisme bawaan untuk mendeteksi bahwa target telah hilang dari frame; classifier selalu mengembalikan bounding box dengan skor tertinggi terlepas dari apakah objek yang benar masih terlihat. Ini adalah alasan utama mengapa drone-seeker mewajibkan lapisan validasi warna eksternal yang dijelaskan di bawah.

### Workaround: Validasi Warna sebagai Pengganti Confidence Score

Karena MIL tidak mengekspos ukuran keyakinan per-frame yang dapat diandalkan, drone-seeker mengimplementasikan deteksi kehilangan dua-lapis yang berjalan paralel dengan MIL setiap frame:

**Lapis 1 — Gate blob warna (per-frame):**
Setelah `TrackerMIL.update()` mengembalikan bounding box `(x, y, w, h)`, pipeline menjalankan `_detection_mask()` (tiga metode deteksi HSV yang sama yang digunakan untuk akuisisi target) pada crop `hsv[y:y+h, x:x+w]`. Jika `cv2.countNonZero(roi_mask) < _MIN_BLOB_AREA` (9 px²), frame ini dihitung sebagai *miss* — MIL tidak dipercaya dan tidak ada sinyal error yang dikirim ke flight controller.

**Lapis 2 — Counter toleransi (`miss_count`):**
Miss tunggal tidak langsung mereset tracker, karena oklusi singkat dan saturasi cahaya sesaat bersifat sementara. Counter `_miss_count` diinkremen setiap frame yang gagal gate warna. Jika `_miss_count >= _KF_MISS_MAX` (5 frame berurutan ≈ 150 ms pada 30 fps), tracker dianggap telah drift sepenuhnya dan direset:

```
_track_win      = None      # lepaskan window tracking
_detect_count   = 0         # kembali ke fase akuisisi
_kf_initialized = False     # reset filter Kalman
_miss_count     = 0
_tracker_obj    = None      # hancurkan objek MIL
```

**Filter Kalman selama periode miss:**
Selama `_miss_count < _KF_MISS_MAX`, filter Kalman tetap berjalan dalam mode *predict-only* (tanpa langkah koreksi pengukuran). Ini memproyeksikan posisi centroid berdasarkan kecepatan yang diestimasi terakhir, sehingga sinyal error yang dikirim ke flight controller tetap kontinu dan halus selama oklusi singkat — bukan lompatan atau gap yang akan menyebabkan osilasi aktuator.

Secara keseluruhan, MIL menyediakan estimasi posisi yang tahan-drift, sedangkan mask HSV menyediakan keputusan biner *ada/tidak-ada objek*. Kombinasi ini menyelesaikan keterbatasan terbesar MIL tanpa memerlukan threshold kepercayaan yang tidak tersedia di API OpenCV.

---

## 6. Morfologi Matematika

Mask biner yang di-threshold mentah dari salah satu dari tiga metode deteksi biasanya mengandung noise salt-and-pepper (piksel positif terisolasi dari kecocokan hue yang tidak disengaja) dan celah intra-objek kecil (piksel yang gagal threshold di dekat tepi atau di region yang teduh). Morfologi matematika, yang fondasi formalnya ditetapkan oleh Serra [12], menyediakan operasi standar untuk membersihkan artefak ini. Opening (erosi diikuti dilatasi dengan elemen penataan yang sama) menghilangkan noise terisolasi tanpa mengikis secara signifikan region terhubung yang besar. Dilatasi selanjutnya memperluas region yang bertahan untuk mengisi lubang kecil dan menggabungkan fragmen yang berdekatan. Drone-seeker menerapkan urutan OPEN → DILATE ini setelah fusi voting mayoritas dari tiga mask deteksi, menggunakan elemen penataan elips 5×5 untuk kedua operasi.

---

## 7. Protokol Komunikasi MAVLink

Sinyal error tracking yang dihitung oleh drone-seeker dikirimkan ke flight controller ArduPlane melalui link serial menggunakan protokol MAVLink. MAVLink diperkenalkan oleh Lorenz Meier di ETH Zurich pada 2009 dan kini dikelola oleh Dronecode Project [14]. Ini adalah protokol framing ringan yang dirancang untuk sistem tertanam dengan sumber daya terbatas: setiap pesan membawa 1-byte CRC extra (konstanta spesifik-tipe-pesan yang dicampur ke dalam checksum) untuk mendeteksi ketidakcocokan versi dan kerusakan pesan tanpa field integritas terpisah. Arsitektur protokol, framing pesan, dan mekanisme CRC-extra disurvei secara komprehensif oleh Koubaa et al. [13], yang juga meninjau integrasi MAVLink ke dalam autopilot ArduPilot dan PX4.

Dalam drone-seeker, error tracking dikirim sebagai pesan `DEBUG_VECT` (MAVLink ID 250), pesan standar yang ada dalam tabel `MAVLINK_MESSAGE_CRCS` yang dikompilasi ArduPlane. Pilihan ini didorong oleh mode kegagalan konkret: `mavlink_get_msg_entry()` MAVLink secara diam-diam membuang pesan apa pun yang ID-nya tidak ada dalam tabel yang dikompilasi, terlepas dari kontennya. Pesan yang dikirim dengan ID non-standar (ID 229, 230, 202 semuanya dicoba) dibuang di penerima sebelum mencapai lapisan aplikasi. Beralih ke pesan `DEBUG_VECT` standar menyelesaikan masalah ini dan memungkinkan penerimaan error tracking yang andal dalam handler mode penerbangan `ModeTracking`.

---

## 8. Kompensasi Latensi dan Panduan Proportional Navigation

### 8.1 Masalah Latensi Pipeline

Setiap loop tracking berbasis gambar digital mengandung penundaan inheren antara saat foton mengenai sensor dan saat sinyal kontrol yang diturunkan mencapai aktuator. Dalam drone-seeker, pipeline ini mencakup eksposur kamera, transfer USB (atau CSI), konversi ruang warna, back-projection, iterasi CamShift, dan serialisasi Python-ke-MAVLink — total diperkirakan sekitar 80 ms dalam kondisi tipikal. Ketika target yang dilacak memiliki kecepatan sudut non-nol relatif terhadap kamera, error yang dihitung dari frame saat ini mewakili sudut LOS pada waktu `t − τ_L`, bukan pada waktu saat ini `t`. Kontroler proporsional yang bertindak berdasarkan pengukuran basi ini memerintahkan koreksi yang secara konsisten tertinggal dari arah target sebenarnya, menghasilkan phase lag yang termanifestasi sebagai osilasi atau offset tracking steady-state.

Solusi klasik untuk loop kontrol dengan time delay murni adalah Smith predictor, yang diperkenalkan oleh Smith [15]. Smith predictor menambah kontroler feedback konvensional dengan model internal plant dan delay, menghasilkan prediksi output pada waktu saat ini dari input masa lalu. Dalam drone-seeker, model plant direduksi menjadi ekstrapolasi Taylor orde pertama dari sudut LOS: beda hingga pengukuran LOS berturut-turut memberikan estimasi laju LOS, yang kemudian digunakan untuk memproyeksikan sudut saat ini ke depan sebesar `τ_L`. Ini memulihkan sudut LOS seketika yang diperkirakan dan menghilangkan phase lag sistematis yang diinduksi oleh pipeline.

### 8.2 Panduan Proportional Navigation

Di luar pemulihan latensi semata, drone-seeker menerapkan lead maju tambahan sebesar `T_PN = 0.30 s` untuk mengimplementasikan hukum panduan **proportional navigation (PN)** yang disederhanakan. Proportional navigation adalah prinsip panduan dominan yang digunakan dalam rudal homing dan dianalisis secara formal oleh Yuan [16] sejak tahun 1948. Dalam bentuk klasiknya, PN memerintahkan akselerasi lateral proporsional terhadap laju sudut LOS:

```
a_c = N · V_c · λ̇
```

di mana `N` adalah konstanta navigasi (biasanya 3–5 untuk implementasi praktis [17]), `V_c` adalah kecepatan penutupan antara pengejar dan target, dan `λ̇` adalah turunan waktu dari sudut LOS `λ`. Properti kunci PN adalah bahwa jika `λ̇ → 0`, pengejar dan target berada pada jalur tabrakan; hukum panduan mendorong pengejar untuk mempertahankan sudut LOS yang konstan, yang merupakan kondisi geometris untuk intersepsi.

Dalam drone-seeker, `errorx` dan `errory` mendekati sudut LOS horizontal dan vertikal di bawah asumsi sudut kecil. Laju beda-hingga mereka `ė_x`, `ė_y` mendekati `λ̇` pada bidang masing-masing. Suku lead PN oleh karena itu mengubah error yang dikirim ke flight controller dari sudut seketika `e(t)` menjadi sudut yang diprediksi `e(t) + ė · T_PN`, yang kemudian didorong ke nol oleh PID ArduPlane. Gain proporsional PID menyerap peran `N · V_c` dalam formulasi klasik, dan `T_PN` adalah satu-satunya yang dapat diatur yang mengontrol agresivitas lead.

Prediktor gabungan — pemulihan latensi ditambah PN lead — setara dengan pendekatan **panduan zero-effort-miss (ZEM)** [17]. ZEM adalah prediksi jarak miss jika pengejar dan target mempertahankan trajektori mereka saat ini; panduan PN memerintahkan akselerasi yang mengurangi ZEM ke nol. Di bawah asumsi kecepatan penutupan konstan dan laju LOS yang berubah lambat selama cakrawala prediksi, ZEM mereduksi menjadi `λ̇ · T_go · V_c`, dan mendorongnya ke nol dengan memerintahkan `e_predicted = e + ė · (τ_L + T_PN) → 0` konsisten dengan panduan PN. Regime throttle penuh fase terminal mempersingkat time-to-go aktual, membuat pendekatan ini lebih akurat saat jangkauan target berkurang.

Perlakuan komprehensif tentang varian proportional navigation — pure PN, true PN, augmented PN — dan sifat optimalitasnya dalam asumsi yang berbeda tentang kemampuan manuver target diberikan oleh Shneydor [17] dan Zarchan [18]. Zarchan khususnya menurunkan metode adjoint untuk analisis sensitivitas jarak miss, yang memberikan landasan teoritis untuk memilih `N` dan menilai dampak latensi dan noise seeker terhadap akurasi intersepsi.
