import glob
import cv2
import numpy as np
import csv
import os
from scipy.spatial import distance as jarak
import argparse

namaFile = "TrainingIndexv2RGB.csv"
folderDataset = "dataset"
binsHSV = (8, 12, 3)  # Pada HSV, 8-> hue, 12-> saturation, 3->value. Total fitur 8 x 12 x 3 = 288
# Kalau BGR, 8->Blue, 12->Saturation, 3->value
binBGR = (9, 9, 9)  # 9->Blue, 9->Saturation, 9->value, total 729
query = cv2.imread("query.jpg")


# appr = argparse.ArgumentParser(description="Aplikasi CBIR dengan histogram")
# appr.add_argument("-t", "--training", help="Melakukan fase training")
# appr.add_argument("-u", "--uji", required=False, type=str, help="Melakukan uji coba dengan query")
#
# args = vars(appr.parse_args())


def training_csv(training=False):
    if (training == True):
        # if (os.path.isfile(namaFile) == False):
        output = open(namaFile, "w")
        print("Proses Training")
        for pathCitra in glob.glob(folderDataset + "/*.jpg"):
            idCitra = pathCitra[pathCitra.rfind("/") + 1:]
            citra = cv2.imread(pathCitra)
            print(pathCitra)

            # Mendeskripsikan citra
            fitur = deskriptor(citra, binBGR)

            # Menulis features yang telah di ambil ke dalam file
            fitur = [str(f) for f in fitur]

            # Fungsi join untuk menggabungkan sequence/array/list dari string menjadi string
            # utuh yang di pisah dengan karakter. Disini di pisah dengan koma
            output.write("%s,%s\n" % (idCitra, ",".join(fitur)))
        print("Training Selesai")
        # Selalu ketika membuka file maka harus di tutup sehingga apa yang sudah di tulis
        # dapat tersimpan di dalamnya
        output.close()
    else:
        print("File sudah ada")


def deskriptor(image, bins):
    citraHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fitur = []

    # Mengambil dimensi dan menghitung titik tengah dari citra
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))  # Titik tengah
    (qX1, qx2) = (int(cX * 0.5), int((w - cX) * 0.5))
    (qY1, qY2) = (int(cY * 0.5), int((h - cY) * 0.5))

    # Membagi citra menjadi 4 segiempat/segment(atas-kiri, atas-kanan,
    # bawah-kanan, bawah-kiri )
    # segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
    segments = [
        # Baris 1
        (0, qX1, 0, qY1), (qX1, cX, 0, qY1), (cX, qx2, 0, qY1), (qx2, w, 0, qY1),
        # Baris 2
        (0, qX1, qY1, cY), (qX1, cX, qY1, cY), (cX, qx2, qY1, cY), (qx2, w, qY1, cY),
        # Baris 3
        (0, qX1, cY, qY2), (qX1, cX, cY, qY2), (cX, qx2, cY, qY2), (qx2, w, cY, qY2),
        # Baris 4
        (0, qX1, qY2, h), (qX1, cX, qY2, h), (cX, qx2, qY2, h), (qx2, w, qY2, h)]
    # Perulangan untuk masing-masing segment
    for (startX, endX, startY, endY) in segments:
        cornerMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

        hist = histogram(image, cornerMask, bins)
        fitur.extend(hist)

    return fitur


def histogram(image, mask, bins):
    # SYNTAX: cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    # images = citra, typenya uint8 atau float32. Harus pake bracket kotak []
    # channels = index dari channel dari citra. Semisal [0] berarti
    # channel yang di ambil itu grayscale. Kalau [0,1,2] yang diambil
    # berarti biru, hijau, merah
    # mask = mask untuk citra. kalau mau cari histogram dari semua
    # citra, pakai None. Tapi kalo pakai region, pakai mask yang
    # sudah di buat
    # histSize = jumlah bins yang dihitung. kalau dihitung semuanya pakai [256]
    # range = rentang. Biasanya menggunakan [0,256]
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    print(image, hist)
    # Melakukan normalisasi pada histogramnya
    hist = cv2.normalize(hist, hist).flatten()
    print("flatten: ", hist)
    return hist


def find_image(indexPath, fiturQuery, limit=10):
    hasil = {}
    with open(indexPath) as file:
        # Inisialisasi CSV Reader
        reader = csv.reader(file)
        for baris in reader:
            # Setiap nilai di parse menjadi float, disimpan
            # dalam bentuk array/list
            features = [float(x) for x in baris[1:]]
            # Melakukan komputasi/perhitungan chi-squared
            # distance dari masing-masing fitur dengan
            # fitur dari query
            # d = self.chi2_distance(features, queryFeatures)
            d = euclidean_distance(features, fiturQuery)

            # Result dictionary-> key: ID citra,
            # value: jarak/distance
            hasil[baris[0]] = d

        file.close()

        # Melakukan sorting dari result dari kecil ke besar
        # Semakin kecil semakin relevan
    hasil = sorted([(v, k) for (k, v) in hasil.items()])

    # Hasil result banyak, tapi di batasi menjadi 10 citra
    # saja/ sesuai dengan limit yang ditentukan
    return hasil[:limit]


# Rumus jarak menggunakan euclidean distance menggunakan scipy
def euclidean_distance(histA, histB):
    d = jarak.euclidean(histA, histB)
    print("jarak: ", d)
    return d


fiturQuery = deskriptor(query, binBGR)

hasilTraining = training_csv(True)
hasil = find_image(namaFile, fiturQuery)

# Menampilkan citra query
cv2.imshow("Query", query)
# querycitra = cv2.resize(query, (0, 0), fx=0.25, fy=0.25)

print("Proses mencari")
# Melakukan perulangan pada hasil
for (score, resultID) in hasil:
    # Menload citra dan menampilkannya
    result = cv2.imread(resultID)
    print(resultID)

    # Meresize ukuran citra hasil
    # result = cv2.resize(result, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
