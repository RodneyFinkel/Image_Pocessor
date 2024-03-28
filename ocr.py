import json
import os
import torch
from paddleocr import PaddleOCR
from transformers import pipeline
import pprintpp

ocr = PaddleOCR(use_angle_cls=True, lang='en',use_space_char=True,show_log=False,enable_mkldnn=True)

img_path = 'data/singapore.jpg'
result = ocr.ocr(img_path, cls=True)

ocr_string = ""
# Extract the text from the OCR result and concatenate it to ocr_string
for i in range(len(result[0])):
    ocr_string = ocr_string + result[0][i][1][0] + " "

# pprintpp.pprint(ocr_string)

model_name = "HuggingFaceH4/zephyr-7b-alpha"
print(model_name)
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
print("checking pipeline function")

# Each message can have 1 of 3 roles: "system" (to provide initial instructions), "user", or "assistant". For inference, make sure "user" is the role in the final message.
messages = [
    {
        "role": "system",
        "content": "You are a JSON converter which receives raw boarding pass OCR information as a string and returns a structured JSON output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the name of the passenger, name of the airline, Flight number, City of Departure, City of Arrival, Date of Departure from this OCR data: {ocr_string}"},
]
# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

print("prompt message dict defined")
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("post prompt call")
outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print("post outputs call")
print(outputs[0]["generated_text"])

# create a JSON object
json_data = {
    "ocr_data": ocr_string,
    "generated_text": outputs[0]["generated_text"]
}

# define the path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), "data", "output.json")

# Write JSON data to file
with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file)

print(f"JSON file save to: {json_file_path}")

