import torch
from paddleocr import PaddleOCR
from transformers import pipeline
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)

# Path to the image file for OCR processing
img_path = 'data/singapore.jpg'

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)

# Extract the text from the OCR result and concatenate it to ocr_string
ocr_string = ""
for i in range(len(result[0])):
    ocr_string = ocr_string + result[0][i][1][0] + " "

# Print the OCR result
print("OCR String:", ocr_string)

# Initialize Google Sheets credentials
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('your_credentials.json', scope)
client = gspread.authorize(creds)

# Access the Google Spreadsheet
spreadsheet = client.open('YourSpreadsheetName')

# Select the first worksheet
worksheet = spreadsheet.sheet1

# Clear existing content in the worksheet
worksheet.clear()

# Write the OCR string to the first cell of the worksheet
worksheet.update('A1', ocr_string)

# Save the OCR string to a text file
with open('ocr_text.txt', 'w') as file:
    file.write(ocr_string)