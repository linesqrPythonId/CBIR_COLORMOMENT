# CBIR_COLORMOMENT
Projek CBIR menggunakan pendekatan warna dengan metode Color Moments. Color space yang digunakan adalah HSV. Dapat di bandingkan dengan RGB

PENGUJIAN: 9389 citra, 9 kategori

color moment:
1. RECALL: 0.03%
2. PRECISION: 30%
3. TRAINING: 21.30 - 14.15 (kira-kira segitu)

Histogram RGB, 16 block 9-bins:
1. RECALL: 0.01%
2. PRECISION: 10%
3. TRAINING: +- 1 jam

Histogram HSV 4 block, 8-bin hue, 12-bin saturation, 3-bin value
1. RECALL: 0.01%
2. PRECISION: 10%
3. TRAINING: +- 2 jam

Akurasinya masih buruk. Yang perlu di perbaiki
1. Untuk color moment perlu menggunakan/perbaikan bobot pada rumus distance. 
2. Untuk histogram menggunakan perbaikan perhitungan jarak atau color spacenya diganti
