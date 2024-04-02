import json
import os
import re
from paddleocr import PaddleOCR
import pprintpp

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)

img_path = 'data/IMG_3503.jpg'
result = ocr.ocr(img_path, cls=True)

ocr_string = ""
# Extract the text from the OCR result and concatenate it to ocr_string

# for i in range(len(result[0])):
#     ocr_string = ocr_string + result[0][i][1][0] + " "

ocr_string = " ".join([word[1][0] for line in result for word in line])

pprintpp.pprint(ocr_string)


