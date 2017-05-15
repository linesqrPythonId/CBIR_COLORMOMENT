import glob
import cv2
import math
import numpy as np
import csv
import os
from scipy.spatial import distance as jarak
import argparse

namaFile = "TrainingIndex2.csv"
folderDataset = "dataset"
bobot = (1, 2, 1)
query = cv2.imread("query.jpg")

def training_csv(training=False):
    index = 0
    if (training == True):
        output = open(namaFile, "w")
        print("Proses Training")
        for pathCitra in glob.glob(folderDataset + "/*.jpg"):
            index += 1
            print("index: ", index, " path: ", pathCitra)
            print("..")
            # Untuk stoppping iterasi
            if (index >= 50):
                break
            idCitra = pathCitra[pathCitra.rfind("/") + 1:]
            citra = cv2.imread(pathCitra)
            citra = resize_image(citra)
            citra = cv2.cvtColor(citra, cv2.COLOR_BGR2HSV)
            # print(pathCitra)
            E = deskriptor_moment1(citra)
            std = deskriptor_moment2(citra, E)
            skw = deskriptor_moment3(citra, E)
            fitur = [E.get(0), E.get(1), E.get(2), std.get(0), std.get(1), std.get(2), skw.get(0), skw.get(1),
                     skw.get(2)]
            fitur = [str(f) for f in fitur]

            output.write("%s,%s\n" % (idCitra, ",".join(fitur)))
        print("Training Selesai")
        output.close()
    else:
        print("File sudah ada")


def find_image(query, indexPath, limit=10):
    print("")
    hasil = {}
    fiturquery = desktiptor_semua(query)

    with open(indexPath) as file:
        reader = csv.reader(file)
        for baris in reader:
            iiop = [0, 1, 2]
            i = iter(iiop)
            # print("NICE")
            # fitur = [float(x) for x in baris[1:]]
            moment1 = [float(x) for x in baris[1:4]]
            mo1 = iter(moment1)
            moment1 = dict(zip(i, mo1))
            print(moment1)

            i = iter(iiop)

            moment2 = [float(x) for x in baris[4:7]]
            mo2 = iter(moment2)
            moment2 = dict(zip(i, mo2))
            print(moment2)

            i = iter(iiop)

            moment3 = [float(x) for x in baris[7:10]]
            mo3 = iter(moment3)
            moment3 = dict(zip(i, mo3))
            print(moment3)
            # fitur = [moment1[0], moment1[1], moment1[2], moment2[0], moment2[1], moment2[2], moment3[0], moment3[1],
            #          moment3[2]]
            fitur = {}

            fitur = [moment1, moment2, moment3]

            # fitur[0][0] = moment1[0]
            # fitur[0][1] = moment1.index(1)
            # fitur[0][2] = moment1.index(2)
            #
            # fitur[1][0] = moment2.index(0)
            # fitur[1][1] = moment2.index(1)
            # fitur[1][2] = moment2.index(2)
            #
            # fitur[2][0] = moment3.index(0)
            # fitur[2][1] = moment3.index(1)
            # fitur[2][2] = moment3.index(2)

            d = distance_moment(fitur, fiturquery, bobot)

            hasil[baris[0]] = d

            # print(moment3)
            # print(fitur)
        file.close()
        # Melakukan sorting dari result dari kecil ke besar
        # Semakin kecil semakin relevan
    hasil = sorted([(v, k) for (k, v) in hasil.items()])

    return hasil[:limit]


def resize_image(img):
    resized_image = cv2.resize(img, (100, 100))
    return resized_image


# Perlu di normalisasi?
def deskriptor_moment1(citra):
    height, width, channel = citra.shape
    # print("height: ", height)
    # print("width: ", width)
    # print("Channel: ", channel)
    E = {}
    E[0] = 0
    E[1] = 0
    E[2] = 0
    # asumsi number of pixel = hxw
    numberOfPixel = height * width
    # print("Number of pixel: ", numberOfPixel)
    try:
        for ch in range(0, channel):
            print(".")
            for xAxis in range(0, height - 1):
                for yAxis in range(0, width - 1):
                    pixel = citra[xAxis, yAxis, ch]
                    E[ch] += pixel * (1 / numberOfPixel)
    except Exception as error:
        print("Error: ", str(error))
    return E


def deskriptor_moment2(citra, E):
    height, width, channel = citra.shape
    numberOfPixel = height * width
    # print("Mean: ", E)
    std = {}
    std[0] = 0
    std[1] = 0
    std[2] = 0
    try:
        for ch in range(0, channel):
            print(".")
            for xAxis in range(0, height - 1):
                for yAxis in range(0, width - 1):
                    pixel = citra[xAxis, yAxis, ch]
                    std[ch] += math.pow((pixel - E[ch]), 2)
            std[ch] = math.sqrt(numberOfPixel * std[ch])
            # print("ch: ", ch, " std: ", std[ch])
    except Exception as error:
        print("Error: ", str(error))
    return std


def deskriptor_moment3(citra, E):
    height, width, channel = citra.shape
    numberOfPixel = height * width
    # print("Mean: ", E)
    skw = {}
    skw[0] = 0
    skw[1] = 0
    skw[2] = 0
    try:
        for ch in range(0, channel):
            print(".")
            for xAxis in range(0, height - 1):
                for yAxis in range(0, width - 1):
                    pixel = citra[xAxis, yAxis, ch]
                    skw[ch] += math.pow((pixel - E[ch]), 3)
            skw[ch] = math.sqrt(numberOfPixel * skw[ch])
            # print("ch: ", ch, " skw: ", skw[ch])
    except Exception as error:
        print("Error: ", str(error))
    return skw


def desktiptor_semua(citra):
    E = deskriptor_moment1(citra)
    std = deskriptor_moment2(citra, E)
    skw = deskriptor_moment3(citra, E)
    # fitur = [E.get(0), E.get(1), E.get(2), std.get(0), std.get(1), std.get(2), skw.get(0), skw.get(1),
    #          skw.get(2)]
    fitur = [E, std, skw]
    return fitur


def distance_moment(deskriptor1, deskriptor2, bobot):
    print("1:->> ", deskriptor1)
    print("2:->> ", deskriptor2)
    dmom = 0
    for ch in range(0, 2):
        dmom += bobot[0] * math.fabs(deskriptor1[0][ch] - deskriptor2[0][ch]) + bobot[1] * math.fabs(
            deskriptor1[1][ch] - deskriptor2[1][ch]) + bobot[2] * math.fabs(deskriptor1[2][ch] - deskriptor2[2][ch])
        # for xAxis in range(0, height - 1):
        #         for yAxis in range(0, width - 1):
    return dmom


# img = cv2.imread("query.jpg")
# resid = resize_image(img)
# dek1 = deskriptor_moment1(resid)
# dek2 = deskriptor_moment2(resid, dek1)
# dek3 = deskriptor_moment3(resid, dek1)
# deskriptor_i = [dek1, dek2, dek3]
# # print(deskriptor_i)
#
# img = cv2.imread("query2.jpg")
# resid = resize_image(img)
# dek1 = deskriptor_moment1(resid)
# dek2 = deskriptor_moment2(resid, dek1)
# dek3 = deskriptor_moment3(resid, dek1)
# deskriptor_h = [dek1, dek2, dek3]
# bobot = (1, 2, 1)
# dmomo = distance_moment(deskriptor_i, deskriptor_h, bobot)
# print("jarak: ", dmomo)

training_csv(True)
query_kecil = resize_image(query)
query_kecil = cv2.cvtColor(query_kecil, cv2.COLOR_BGR2HSV)
hasil = find_image(query_kecil, namaFile)

for (score, resultID) in hasil:
    result = cv2.imread(resultID)
    print(resultID)
    result = cv2.resize(result, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("result", result)
    cv2.waitKey(0)



# for kl in range(0, 100):
#
#     print(kl)
