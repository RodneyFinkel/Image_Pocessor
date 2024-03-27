import torch
from transformers import AutoModelForCausalLM, pipeline
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, show_log=False, enable_mkldnn=True)
img_path = 'data/singapore.jpg'
model_name = "HuggingFaceH4/zephyr-7b-alpha"

# Define the function to save model state dictionary to disk
def save_model_to_disk(model, file_path):
    torch.save(model.state_dict(), file_path)

# Define the function to load model state dictionary from disk
def load_model_from_disk(model, file_path):
    model.load_state_dict(torch.load(file_path))

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model state dictionary to disk
save_model_to_disk(model, "model_state_dict.pth")

# Load the model state dictionary from disk
loaded_model = AutoModelForCausalLM.from_pretrained(model_name)
load_model_from_disk(loaded_model, "model_state_dict.pth")

# Create pipeline using the loaded model
pipe = pipeline("text-generation", model=loaded_model, torch_dtype=torch.bfloat16, device_map="auto")

# Define the function to process image and generate JSON
def Image_to_JSON(img_path):
    try:
        result = ocr.ocr(img_path, cls=True)
        ocr_string = ""
        for i in range(len(result[0])):
            ocr_string = ocr_string + result[0][i][1][0] + " "

        print("OCR Result:", ocr_string)

        messages = [
            {"role": "system", "content": "You are a JSON converter which receives raw boarding pass OCR information as a string and returns a structured JSON output by organising the information in the string."},
            {"role": "user", "content": f"Extract the name of the passenger, name of the airline, Flight number, City of Departure, City of Arrival, Date of Departure from this OCR data: {ocr_string}"},
        ]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(outputs[0]["generated_text"])
        return outputs[0]["generated_text"]
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    Image_to_JSON(img_path)
