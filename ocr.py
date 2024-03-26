import torch
from paddleocr import PaddleOCR
from transformers import AutoModelForCausalLM, pipeline

ocr = PaddleOCR(use_angle_cls=True, lang='en',use_space_char=True,show_log=False,enable_mkldnn=True)
img_path = 'data/singapore.jpg'
model_name = "HuggingFaceH4/zephyr-7b-alpha"

# Using the diskoffload function:
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto")


# Each message can have 1 of 3 roles: "system" (to provide initial instructions), "user", or "assistant". 
# For inference, make sure "user" is the role in the final message.

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating


def Image_to_JSON(image_path):
    # Perform OCR on the image and extract the text content
    result = ocr.ocr(image_path, cls=True)

    ocr_string = ""  # Stores the OCR content extracted from the image in a string which can be fed into ChatGPT

    # Extract the text from the OCR result and concatenate it to ocr_string
    for i in range(len(result[0])):
        ocr_string = ocr_string + result[0][i][1][0] + " "

    messages = [
    {
        "role": "system",
        "content": "You are a JSON converter which receives bank statement OCR information as a string and returns a structured JSON output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the name of the passenger, name of the airline, Flight number, City of Departure, City of Arrival, Date of Departure from this OCR data: {ocr_string}"},
    ]
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]