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

patterns = {
    "bank_name": r"www\.([^.]+)\.co\.uk",  # Extracts the bank name from the website URL
    "account_holder": r"ames\s([A-Za-z\s]+)\sVIS\sDRP",  # Extracts the account holder's name
    "transactions": r"(\b(?:EUR|USD|GBP)\s[\d,.]+\b)"  # Extracts transactions
}


# Extracted information dictionary
extracted_info = {}

# Find matches using the patterns
for key, pattern in patterns.items():
    match = re.search(pattern, ocr_string)
    if match:
        extracted_info[key] = match.group(1).strip()

# Extract transactions
transactions = re.findall(patterns["transactions"], ocr_string)
extracted_info["transactions"] = [{"transaction": t, "description": ""} for t in transactions]

# Create a JSON object with the extracted information
json_data = extracted_info

# Define the path to the JSON file
json_file_path = os.path.join(os.getcwd(), "data", "output.json")

# Write JSON data to file
with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

print(json.dumps(json_data, indent=4))
print(f"JSON file saved to: {json_file_path}")
