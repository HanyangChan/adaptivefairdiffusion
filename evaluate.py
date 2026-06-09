import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class Evaluator:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    @torch.no_grad()
    def compute_clip_score(self, image: Image.Image, prompt: str):
        """
        Computes the CLIP score (cosine similarity) between the image and the prompt.
        Higher is better.
        """
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        
        # image_embeds and text_embeds
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        cosine_similarity = (image_embeds @ text_embeds.T).item()
        # Scale to match standard clip score scale (usually 100 * similarity)
        return cosine_similarity * 100.0

    @torch.no_grad()
    def evaluate_fairness(self, image: Image.Image, attributes: list):
        """
        Predicts the probability distribution over a list of attributes (e.g., ["a photo of a man", "a photo of a woman"]).
        Returns a dict of attribute -> probability.
        """
        inputs = self.processor(text=attributes, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
        
        return {attr: float(prob) for attr, prob in zip(attributes, probs)}

if __name__ == '__main__':
    # test
    # e = Evaluator()
    # img = Image.new('RGB', (224, 224))
    # print(e.compute_clip_score(img, "a photo of a dummy"))
    # print(e.evaluate_fairness(img, ["a man", "a woman"]))
    pass
