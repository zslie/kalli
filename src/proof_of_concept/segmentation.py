import json
import cv2
import pytesseract
import time
from math import floor

def segment_letters_tesseract_cv2(image_file):
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


def filter_words(json_doc):
    return [item for item in json_doc["Blocks"] if item['BlockType'] == "WORD"]

def self_authored_segmentation(image_file_path, json_file_path):
    start = time.time()

    # load image file
    img = cv2.imread(image_file_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img_gray.shape

    # load JSON file
    with open(json_file_path) as jf:
        aws_json = json.load(jf)

    json_word_array = filter_words(aws_json)
    print(json_word_array)

    # for word in json
    #   for letter in word
    #       detect until highest value for letter, 
    #       moving "greedily" while keeping account for the next letter
    for json_word in json_word_array:
        word_str = json_word["Text"]
        bl, br, tr, tl = json_word["Geometry"]["Polygon"] 
        bl_px = { 'X': floor(bl['X'] * img_width), 'Y': floor(bl['Y'] * img_height) }
        br_px = { 'X': floor(br['X'] * img_width), 'Y': floor(br['Y'] * img_height) }
        tr_px = { 'X': floor(tr['X'] * img_width), 'Y': floor(tr['Y'] * img_height) }
        tl_px = { 'X': floor(tl['X'] * img_width), 'Y': floor(tl['Y'] * img_height) }
        # print(bl)
        # print("img_height: " + str(img_height) + " img_width: " + str(img_width))
        # print(bl)
        # print(bl_px)

        # The format is img[y_start:y_end, x_start:x_end]
        word_roi = img_gray[bl_px['Y']:tl_px['Y'], bl_px['X']:br_px['X']]
        word_roi_h, word_roi_w = word_roi.shape
        # cv2.imshow("Region of Interest", word_roi)
        # cv2.waitKey(0)

        letters = list(word_str)
        letter_count = len(letters)
        previous_letter_start_x = 0
        previous_letter_end_x = 0
        segment_size_increase = 20 # todo - make this dynamic/computed
        for letter in letters:
            confidence = 0.0
            confidence_scores = []
            while(previous_letter_start_x < word_roi_w):
                previous_letter_start_x += segment_size_increase
                letter_roi = word_roi[:, previous_letter_end_x:previous_letter_start_x]
                # Run OCR
                psm_single_char_mode = "10"
                config = f"-l eng --oem 1 --psm {psm_single_char_mode} -c hocr_char_boxes=1"
                data = pytesseract.image_to_data(letter_roi, config=config)
                text = pytesseract.image_to_string(letter_roi, config=config)
                print(data)
                print("Detected text:", text)
                if (text == letter):
                    continue

            return # todo remove

            print("Detected letter: " + letter + " with confidence: " + confidence)

    end = time.time()
    print("Time taken to run: " + str(end-start) + " seconds")

image_file = '../sample/aws-textract-1/a01-007u-s02-02.png'
json_file  = '../sample/aws-textract-1/analyzeDocResponse.json'
# segment_letters_tesseract_cv2(image_file)
self_authored_segmentation(image_file, json_file)


def parseJson():
    print("unimplemented!")


# https://www.cse.sc.edu/~songwang/document/wacv13c.pdf
def segmentLetters():
    print("unimplemented!")



