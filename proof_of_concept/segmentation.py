import json


import cv2
import pytesseract
import time

def segment_letters(image_file):
    print("PTSS Version: " + str(pytesseract.get_tesseract_version()))
    start = time.time()

    # pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' # change this to the location where Tesseract is installed

    # Load image
    img = cv2.imread(image_file)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape


    # Use Tesseract to do OCR on the image and get bounding boxes
    boxes = pytesseract.image_to_boxes(img_gray)

    # Create bounding box on image and save file:
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     char = b[0]
    #     x_start, y_start, x_end, y_end = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
    #     cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    # cv2.imwrite("../sample/aws-textract-1/box-me-up.png", img)

    # For each detected character, print the character and its bounding box coordinates
    for b in boxes.splitlines():
        b = b.split(' ')
        char = b[0]
        x_start, y_start, x_end, y_end = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        print(f"Character: {char}, Start: ({x_start}, {y_start}), End: ({x_end}, {y_end})")

    end = time.time()
    print("Time taken to run: " + str(end-start) + " seconds")

image_file = '../sample/aws-textract-1/a01-007u-s02-02.png'
segment_letters(image_file)



def parseJson():
    print("unimplemented!")


# https://www.cse.sc.edu/~songwang/document/wacv13c.pdf
def segmentLetters():
    print("unimplemented!")



