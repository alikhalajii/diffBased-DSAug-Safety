import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import  StableDiffusionPipeline_LEDITS, StableDiffusionPipelineXL_LEDITS
from helper_functions import load_image, show_image_grid


# ImageEditor class to handle the image editing process
class ImageEditor:
    def __init__(self, model_name, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.pipeline = self.load_leditspp(model_name)
    
    def load_leditspp(self, model):
        """ Load the LEDITS++ model based on the model name and device. """
        #if model == 'runwayml/stable-diffusion-v1-5': # Model deprecated
        if model == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
            pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model, safety_checker=None)
            pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(model, subfolder="scheduler", 
                                                                               algorithm_type="sde-dpmsolver++", solver_order=2)
        elif model == 'stabilityai/stable-diffusion-xl-base-1.0':
            pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(model, safety_checker=None)
            pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(model, subfolder="scheduler",
                                                                              algorithm_type="sde-dpmsolver++", solver_order=2)
        else:
            raise ValueError("Invalid model name")
        
        pipe.to(self.device)
        return pipe

    def edit_image(self, image, guidance_terms):
        """ Choose the correct pipeline based on the model name. """
        if self.model_name == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
            if len(guidance_terms) > 1:
                return self.ledits_pipeline_duo(image, guidance_terms)
            else:
                return self.ledits_pipeline_15(image, guidance_terms)
        elif self.model_name == 'stabilityai/stable-diffusion-xl-base-1.0':
            return self.ledits_pipeline_xl(image, guidance_terms)
        else:
            raise ValueError("Invalid model name")

    def ledits_pipeline_15(self, image, safety_guidance):
        """ Apply stable_diff1.5 pipeline with given guidance terms. """
        skips = [0.3, 0.2, 0.15]
        edit_thresholds = [0.9, 0.8, 0.75]
        guidance_scales = [10, 15]
        seed = 1

        total_steps = len(skips) * len(edit_thresholds) * len(guidance_scales)

        org = load_image(image).resize((512,512))
        im = np.array(org)[:, :, :3]

        images = []
        parameter_grid = []

        with tqdm(total=total_steps, disable=True) as pbar:
            for skip in skips:
                for edit_threshold in edit_thresholds:
                    for guidance_scale in guidance_scales:
                        gen = torch.manual_seed(seed)
                        with torch.no_grad():
                            _ = self.pipeline.invert(im, num_inversion_steps=50, generator=gen, verbose=False, skip=skip)
                            out = self.pipeline(editing_prompt=safety_guidance, 
                                                edit_threshold=[edit_threshold],
                                                edit_guidance_scale=[guidance_scale],
                                                reverse_editing_direction=[True],
                                                use_intersect_mask=True,)
                        
                        images.append(out.images[0])
                        params = {"seed": seed, "skip": skip, "edit_threshold": edit_threshold, "guidance_scale": guidance_scale, "image": out.images[0]}                   
                        parameter_grid.append(params)
                        
                        #display(out.images[0])
                        pbar.update(1)
        return parameter_grid

    def ledits_pipeline_duo(self, image, safety_guidance):
        """ Apply stable_diff1.5 with both neg and pos terms. """
        skips = [0.2, 0.15]
        edit_thresholds = [[.9, .85], [.8, .8], [0.75, .9]]
        guidance_scales = [[6, 3], [10, 5], [15, 5], [12, 8]]
        seed = 1

        total_steps = len(skips) * len(edit_thresholds) * len(guidance_scales)

        org = load_image(image).resize((512,512))
        im = np.array(org)[:, :, :3]

        images = []
        parameter_grid = []

        with tqdm(total=total_steps, disable=True) as pbar:
            for skip in skips:
                for edit_threshold in edit_thresholds:
                    for guidance_scale in guidance_scales:
                        gen = torch.manual_seed(seed)
                        with torch.no_grad():
                            _ = self.pipeline.invert(im, num_inversion_steps=50, generator=gen, verbose=False, skip=skip)
                            out = self.pipeline(editing_prompt=safety_guidance, 
                                                edit_threshold=edit_threshold,
                                                edit_guidance_scale=guidance_scale,
                                                reverse_editing_direction=[True, False],
                                                use_intersect_mask=True,)
                        
                        images.append(out.images[0])
                        params = {"seed": seed, "skip": skip, "edit_threshold": edit_threshold, "guidance_scale": guidance_scale, "image": out.images[0]}                   
                        parameter_grid.append(params)
                        
                        pbar.update(1)
        #show_image_grid(parameter_grid, image_key='image', title_keys=['seed', 'skip', 'edit_threshold', 'guidance_scale'], grid_size=(4, 6))
        return parameter_grid

    def ledits_pipeline_xl(self, image_file, safety_guidance):
        """ Apply stable_diff_xl pipeline with given guidance terms. """
        img = Image.open(image_file)
        org = load_image(img.resize((1024,1024)))

        seed = 1
        skip = 0.3
        edit_threshold = 0.9
        guidance_scales = [10, 15]
        edit_warmup_steps = [4, 6]

        total_steps = len(edit_warmup_steps) * len(guidance_scales)

        images = []
        parameter_grid = []

        with tqdm(total=total_steps, disable=True) as pbar:
            for warmup_steps in edit_warmup_steps:
                for guidance_scale in guidance_scales:
                    gen = torch.manual_seed(seed)
                    with torch.no_grad():
                        _ = self.pipeline.invert(image_file, num_inversion_steps=100, generator=gen, verbose=False, skip=skip)
                        out = self.pipeline(editing_prompt=safety_guidance, 
                                            edit_threshold=[edit_threshold],
                                            edit_guidance_scale=[guidance_scale], 
                                            reverse_editing_direction=[True],
                                            edit_warmup_steps=[warmup_steps],
                                            use_intersect_mask=True)
                        
                        images.append(out.images[0])
                        parameter_grid.append({"seed": seed, "skip": skip, "edit_threshold": edit_threshold, "guidance_scale": guidance_scale, "warmup_steps": warmup_steps, "image": out.images[0]})
                        pbar.update(1)
        return parameter_grid
