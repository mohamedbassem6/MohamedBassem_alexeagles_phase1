import cv2 as cv
import utils


ideal_img = cv.imread("samples/ideal.jpg")
if ideal_img is None:
    print("Ideal image not found")
    exit()

_, ideal_img_bin = cv.threshold(cv.cvtColor(ideal_img, cv.COLOR_BGR2GRAY), utils.THRESH_VAL, 255, cv.THRESH_BINARY)

sample_img_path = input("Enter image path: ")
sample_img = cv.imread(sample_img_path)
if sample_img is None:
    print("Sample image not found")
    exit()
    
_, sample_img_bin = cv.threshold(cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY), utils.THRESH_VAL, 255, cv.THRESH_BINARY)

diff = utils.get_teeth_diff(ideal_img_bin, sample_img_bin)
contours_count = utils.get_contours_count(diff)

inner_area = utils.get_inner_area(sample_img_bin)


RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

print("Broken/Worn Teeths: ", end="")
if contours_count > 0:
    print(RED + str(contours_count))
else:
    print(GREEN + "None")
print(RESET, end="")

print("Inner Opening: ", end="")
if inner_area < utils.IDEAL_INNER_AREA:
    print(RED + "Smaller")
elif inner_area > utils.IDEAL_INNER_AREA:
    print(RED + "Larger")
else:
    print(GREEN + "Identical")
print(RESET, end="")