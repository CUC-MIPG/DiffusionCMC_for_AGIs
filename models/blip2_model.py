from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class ImageCaptioning:
    def __init__(self, device):
        self.device = device
        self.processor, self.model = self.initialize_model()

    def initialize_model(self):

        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=self.data_type
        )
        model.to(self.device)
        return processor, model

    def image_caption(self, ref_image):
        
        inputs = self.processor(images=ref_image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs,max_new_tokens=512)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text