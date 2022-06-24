# -*- coding: utf-8 -*-
"""
@author: lucas.moorlag
"""
import cv2
import csv

def transform_point(img, point):

    img_lat_lt = 6250
    img_lon_lt = -520
    img_lat_rb = 5050
    img_lon_rb = 1310

    img_lat_height = img_lat_lt - img_lat_rb
    img_lon_width = img_lon_rb - img_lon_lt

    width = img.shape[1]
    height = img.shape[0]

    lat = point[1]
    lon = point[0]
    return int(((lon - img_lon_lt) / img_lon_width) * width), int(((img_lat_lt - lat) / img_lat_height) * height) 

"""
point in the for x, y in domain [0, 1]
color in BGR
"""

def draw_line(img, point1, point2, color=(255, 0, 0), thickness=1):
    cv2.line(img, transform_point(img, (point1['lon'], point1['lat'])), transform_point(img, (point2['lon'], point2['lat'])), color=color, thickness=thickness)

def draw_point(img, point, thickness=.1, color=(0, 0, 255)):
    p1 = transform_point(img, (point['lon'] - thickness, point['lat'] - thickness))
    p2 = transform_point(img, (point['lon'] + thickness, point['lat'] + thickness))

    cv2.rectangle(img, p1, p2, color=color, thickness=4)

def parse_harbor_file():
    res = []
    with open("./data_havenlocaties_fugro_3.txt") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for idx, row in enumerate(rd):
            if idx == 0:
                continue
            res.append({
                "campaign": row[0],
                "lat": int(float(row[2].replace(',', '.')) * 100),
                "lon": int(float(row[1].replace(',', '.')) * 100),
            })
    return res

if __name__ == "__main__":
    harbors = parse_harbor_file()
    img_path = './img5.jpg'
    img = cv2.imread(img_path)
    print(len(harbors))
    print(transform_point(img, (harbors[0]['lon'], harbors[0]['lat'])))
    while True:

        for point in harbors:
            draw_point(img, point, color=(0, 0, 255))

        for idx, harbor in enumerate(harbors):
            if idx % 50 == 0:
                for idx2, harbor2 in enumerate(harbors):
                    if idx2 % 50 == 0:
                        draw_line(img, harbor, harbor2)

        cv2.imshow('name', img)
        res = cv2.waitKey()
        # CLOSE BY PRESSING 'q'
        if res == ord('q'):
            cv2.destroyAllWindows()
            break