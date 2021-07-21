from ANPR import find_plate
import cv2


if __name__ == '__main__':
    img = cv2.imread('data/license_plate_A.jpeg')
    plate_number = find_plate(img)
    print('PLATE NUMBER IS:', plate_number)

