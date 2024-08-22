import cv2 as cv
import numpy as np

THRESH_VAL = 30
IDEAL_INNER_AREA = 2108.5

def get_teeth_diff(ideal_img: np.array, sample_img: np.array):
    output_bin = cv.bitwise_xor(ideal_img, sample_img)

    circle = cv.circle(np.ones_like(ideal_img) * 255, (ideal_img.shape[1] // 2, ideal_img.shape[0] // 2), 100, 0, -1)
    output_bin = cv.bitwise_and(output_bin, circle)

    processed_output = cv.erode(output_bin, np.ones((1, 1), np.uint8), iterations=1)
    processed_output = cv.morphologyEx(processed_output, cv.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    return processed_output


def get_inner_area(img: np.array):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    circle_contour = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            circle_contour.append(contour)

    return cv.contourArea(circle_contour[0]) if len(circle_contour) > 0 else 0


def get_contours_count(img: np.array):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)