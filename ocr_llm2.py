import json
import os
import re
import torch
from paddleocr import PaddleOCR
from transformers import pipeline
import pprintpp

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)

img_path = 'data/IMG_3503.jpg'
result = ocr.ocr(img_path, cls=True)

# Initialize variables to store extracted information
bank_name = ""
account_name = ""
sort_code = ""
account_number = ""
date = ""
payment_details = ""

# Define mappings from descriptor to corresponding field
field_mapping = {
    "Bank Name": "bank_name",
    "Account Name": "account_name",
    "Sort Code": "sort_code",
    "Account Number": "account_number",
    "Date": "date",
    "Payment Details": "payment_details"
}

# Extract information based on descriptors
for line in result:
    for word in line:
        for i, char in enumerate(word[1][0]):
            if char == ':':
                descriptor = word[1][0][:i].strip()
                if descriptor in field_mapping:
                    field = field_mapping[descriptor]
                    value = word[1][0][i+1:].strip()
                    locals()[field] = value

# Generate additional text using Hugging Face model
model_name = "HuggingFaceH4/zephyr-7b-beta"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
messages = [
    {
        "role": "system",
        "content": "You are a JSON converter which receives raw bank statement OCR information as a string and returns a structured JSON output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the name of the bank, Account Name, SortCode, Account Number, Date, Payment type and details from this OCR data."},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# Create a JSON object
json_data = {
    "Bank Name": bank_name,
    "Account Name": account_name,
    "Sort Code": sort_code,
    "Account Number": account_number,
    "Date": date,
    "Payment Details": payment_details,
    "Generated Text": outputs[0]["generated_text"]
}

# Define the path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), "data", "output.json")

# Write JSON data to file
with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

print(f"JSON file saved to: {json_file_path}")
