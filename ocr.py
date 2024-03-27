import torch
from paddleocr import PaddleOCR
from transformers import pipeline



ocr = PaddleOCR(use_angle_cls=True, lang='en',use_space_char=True,show_log=False,enable_mkldnn=True)

img_path = 'data/singapore.jpg'
result = ocr.ocr(img_path, cls=True)



ocr_string = ""
# Extract the text from the OCR result and concatenate it to ocr_string
for i in range(len(result[0])):
    ocr_string = ocr_string + result[0][i][1][0] + " "

print(ocr_string)

model = "HuggingFaceH4/zephyr-7b-alpha"
pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto" )
