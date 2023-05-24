import torch
# from lavis.models import load_model_and_preprocess

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class GlobalDescriptor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # # loads BLIP-2 pre-trained model
        # # LAVIS
        # self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=self.device)

        # HuggingFace
        self.vis_processors = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.model.to(self.device)

    def get_global_description(self, image_src):
        image = Image.open(image_src)
        # # LAVIS
        # # BLIP FLAN
        # raw_image = image.convert("RGB")
        # # prepare the image
        # image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # generated_txt = self.model.generate({"image": image, "prompt": "Question: what is in this photo? Answer:"})
        # return generated_txt

        # HuggingFace
        inputs = self.vis_processors(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text