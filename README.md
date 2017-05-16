# CBIR_COLORMOMENT
Projek CBIR menggunakan pendekatan warna dengan metode Color Moments. Color space yang digunakan adalah HSV. Dapat di bandingkan dengan RGB

PENGUJIAN: 9389 citra, 9 kategori

color moment:
RECALL: 0.03%
PRECISION: 30%
TRAINING: 21.30 - 14.15 (kira-kira segitu)

Histogram RGB, 16 block 9-bins:
RECALL: 0.01%
PRECISION: 10%
TRAINING: +- 1 jam

Histogram HSV 4 block, 8-bin hue, 12-bin saturation, 3-bin value

RECALL: 0.01%
PRECISION: 10%
TRAINING: +- 2 jam

Akurasinya masih buruk. Yang perlu di perbaiki
1. Untuk color moment perlu menggunakan/perbaikan bobot pada rumus distance. 
2. Untuk histogram menggunakan perbaikan perhitungan jarak atau color spacenya diganti
