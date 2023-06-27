from model import DonutModel
from util import DonutDataset

model = DonutModel.from_pretrained("naver-clova-ix/donut-base")
dataset = DonutDataset("/data/share/project/entero/pdf_images2", model, max_length=2048, split="test")
print(dataset[0])
