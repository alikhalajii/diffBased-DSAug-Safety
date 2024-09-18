# image_similarity.py

import torch
from PIL import Image
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import clip


class ImageSimilarity:
    def __init__(self, device='cuda'):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    def get_clip_score(self, image1: Image.Image, image2: Image.Image) -> float:
        image1_preprocess = self.clip_preprocess(image1).unsqueeze(0).to(self.device)
        image2_preprocess = self.clip_preprocess(image2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image1_features = self.clip_model.encode_image(image1_preprocess)
            image2_features = self.clip_model.encode_image(image2_preprocess)

        cos = torch.nn.CosineSimilarity(dim=0)
        similarity = cos(image1_features[0], image2_features[0]).item()
        
        return (similarity + 1) / 2

    def get_lpips_score(self, org_image: Image.Image, edited_image: Image.Image) -> float:
        org_image_np = np.array(org_image)[:, :, :3]
        edited_image_np = np.array(edited_image)[:, :, :3]

        org_image_tensor = torch.tensor(org_image_np).permute(2, 0, 1).float() / 127.5 - 1.0
        edited_image_tensor = torch.tensor(edited_image_np).permute(2, 0, 1).float() / 127.5 - 1.0

        org_image_tensor = org_image_tensor.unsqueeze(0)
        edited_image_tensor = edited_image_tensor.unsqueeze(0)

        similarity_score = self.lpips_model(org_image_tensor, edited_image_tensor)
        return similarity_score.item()

    @staticmethod
    def normalize_score(score: float, min_val: float, max_val: float) -> float:
        return (score - min_val) / (max_val - min_val)

    def get_combined_similarity_score(self, clip_score: float, lpips_score: float, alpha=0.7) -> float: 
        combined_score = alpha * clip_score + (1 - alpha) * lpips_score
        #print(f"Combined similarity score: {round(combined_score, 4)}")
        return combined_score
