# pre-requiste libraries for Zephyr-7B.




!pip install git+https://github.com/huggingface/transformers.git
!pip install accelerate


# Pre-requisite libraries for PaddleOCR

!git clone https://github.com/PaddlePaddle/PaddleOCR.git

pip install paddlepaddle
from paddleocr import PaddleOCR

!python3 -m pip install paddlepaddle-gpu
!pip install "paddleocr>=2.0.1"


Reference:
https://medium.com/@sureshraghu0706/from-images-to-insights-3-tiered-data-extraction-from-images-with-ocr-and-large-language-models-0c07754813cc