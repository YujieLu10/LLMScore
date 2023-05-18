import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class GlobalDescriptor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # loads BLIP-2 pre-trained model
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=self.device)

    def get_global_description(self, image_src):
        image = Image.open(image_src)
        # BLIP FLAN
        raw_image = image.convert("RGB")
        # prepare the image
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        generated_txt = self.model.generate({"image": image, "prompt": "Question: what is in this photo? Answer:"})
        return generated_txt