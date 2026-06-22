import cv2 as cv
import numpy as np
import os


def chceck(x, y, x0, y0, x1, y1):
    return x0 < x < x1 and y0 < y < y1


folder_dir = '.'

for image in sorted(os.listdir(folder_dir)):
    if image.endswith('.jpg'):
        print(image)

        img = cv.imread(image, cv.IMREAD_COLOR)

        if img is None:
            print("Nie wczytano:", image)
            continue

        frameColour = (255, 100, 200)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_cir = cv.GaussianBlur(img, (5, 5), 0)
        img_cir = cv.medianBlur(img_cir, 5)
        img_cir = cv.cvtColor(img_cir, cv.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv.filter2D(img_cir, -1, kernel)

        #krawedzie
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100,
                               minLineLength=50, maxLineGap=10)

        if lines is None:
            print("Nie wykryto tacki")
            continue

        x = []
        y = []

        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
            y.append(line[0][1])
            y.append(line[0][3])

        x0, x1 = min(x), max(x)
        y0, y1 = min(y), max(y)

        #wyrysowanie obrebu tacy
        cv.rectangle(img, (x0, y0), (x1, y1), frameColour, 3)

        #okregi monet
        circles = cv.HoughCircles(dst, cv.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=35,
                                 minRadius=20, maxRadius=40)

        if circles is None:
            print("Nie wykryto monet")
            continue

        circles = np.uint16(np.around(circles))

        radiusSize = [i[2] for i in circles[0, :]]

        sumIn = 0
        sumOut = 0
        countIn = 0
        countOut = 0

        for i in circles[0, :]:

            # klasyfikacja monet
            if i[2] >= max(radiusSize) - 3:
                radC = (0, 255, 255)  # 5 zł
                zm = 5
                label = "5 zl"
            else:
                radC = (255, 0, 0)    # 5 gr
                zm = 0.05
                label = "0.05 zl"

            # rysowanie okręgu
            cv.circle(img, (i[0], i[1]), i[2], radC, 2)

            # położenie
            if chceck(i[0], i[1], x0, y0, x1, y1):
                sumIn += zm
                countIn += 1
            else:
                sumOut += zm
                countOut += 1

            # środek
            cv.circle(img, (i[0], i[1]), 2, radC, 3)

            # podpis monety
            cv.putText(
                img,
                label,
                (i[0] - 30, i[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                radC,
                2
            )

        #wyniki
        cv.putText(img, f"Na tacy: {countIn} monet, {round(sumIn, 2)} zl",
                   (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (0, 255, 0),
                   2)

        cv.putText(img, f"Poza taca: {countOut} monet, {round(sumOut, 2)} zl",
                   (10, 65),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (0, 0, 255),
                   2)

        cv.putText(img, f"Suma: {round(sumIn + sumOut, 2)} zl",
                   (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.9,
                   (255, 255, 255),
                   2)

        cv.imshow(image, img)
        cv.waitKey(0)

cv.destroyAllWindows()