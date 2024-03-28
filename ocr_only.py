import json
import os
import re
import torch
from paddleocr import PaddleOCR
from transformers import pipeline

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)

img_path = 'data/singapore.jpg'
result = ocr.ocr(img_path, cls=True)

ocr_string = ""
# Extract the text from the OCR result and concatenate it to ocr_string
for i in range(len(result[0])):
    ocr_string = ocr_string + result[0][i][1][0] + " "

print(ocr_string)

# Define a regular expression pattern to extract relevant information
pattern = r"(\b[A-Za-z0-9\s]+)\b"
pattern2 = r"(\b[A-Z][A-Z\s]+)\b"
pattern3 = r"(\b[A-Z0-9\s]+)\b"
pattern4 = r"((?:\b[A-Z][A-Z\s]+\b)|(?:Flight: ([^\n]+))|(?:From: ([^\n]+))|(?:To: ([^\n]+))|(?:Date: ([^\n]+))|(?:Boarding time: ([^\n]+))|(?:Airline Name: ([^\n]+)))"

# Find matches using the pattern
matches = re.findall(pattern4, ocr_string)

# Extracted information
passenger_name = matches[0]
Flight = matches[1]
From = matches[2]
To = matches[3]
Date = matches[4]
Boarding_Time = matches[5]
airline_name = matches[6]

for match in matches:
    if match[0]:
        passenger_name = match[0]
    elif "Flight:" in match[1]:
        Flight = match[1].split(":")[1].strip()
    elif "From:" in match[1]:
        From = match[1].split(":")[1].strip()
    elif "To:" in match[1]:
        To = match[1].split(":")[1].strip()
    elif "Date:" in match[1]:
        Date = match[1].split(":")[1].strip()
    elif "Boarding Time:" in match[1]:
        Boarding_Time = match[1].split(":")[1].strip()
    elif "Airline Name:" in match[1]:
        airline_name = match[1].split(":")[1].strip()

# Create a JSON object with the extracted information
json_data = {
    "passenger_name": passenger_name,
    "airline_name": airline_name,
    "flight_number": Flight,
    "departure_city": From,
    "arrival_city": To,
    "departure_date": Date,
    "boarding_time": Boarding_Time
 #   "generated_text": outputs[0]["generated_text"]  # Add generated text if needed
}

# Define the path to the JSON file
json_file_path = os.path.join(os.getcwd(), "data", "output.json")

# Write JSON data to file
with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

print(f"JSON file saved to: {json_file_path}")
