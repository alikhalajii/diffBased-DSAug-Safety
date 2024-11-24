Pipeline for Diffusion-Based Dataset Augmentation with Safety in Mind

## Usage


1. For building the running environment:

You can use [SGLang Docker](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile) 
and then in your container terminal:
```bash
git clone https://github.com/alikhalajii/diffBased-DSAug-Safety.git
pip install git+https://github.com/ml-research/ledits_pp.git
pip install git+https://github.com/openai/CLIP.git
``` 

Alternatively, create the Conda environment using the provided YAML file:
```bash
git clone https://github.com/alikhalajii/diffBased-DSAug-Safety.git
cd diffBased-DSAug-Safety
conda env create -f sgl-lpp.yml
conda activate sgl-lpp
``` 


2. Set the Hugging Face token with access to the following repositories:

AIML-TUDA/LlavaGuard-7B,
AIML-TUDA/LlavaGuard-13B,
llava-hf/llama3-llava-next-8b-hf,
llava-v1.6-vicuna-13b-hf

and activate your access token:
```bash
huggingface-cli login
```


3. Reinstall the flashinfer package:
```bash
pip uninstall flashinfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```


4. Launch the SGLang server as backend with one of the below LlavaGuard checkpoints.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path AIML-TUDA/LlavaGuard-7B --tokenizer-path llava-hf/llava-1.5-7b-hf --port 10000
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path AIML-TUDA/LlavaGuard-13B --tokenizer-path llava-hf/llava-1.5-13b-hf --port 10000
```


5. Once the server is set up, run the safeguarding pipeline:
```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_pipe.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --llava_model "llava-hf/llama3-llava-next-8b-hf"
CUDA_VISIBLE_DEVICES=1 python3 src/main_pipe.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --llava_model "llava-hf/llava-v1.6-vicuna-13b-hf"
```
To limit the number of images in the input directory to a specific number (e.g., 100):
```bash
CUDA_VISIBLE_DEVICES=1 python src/main_pipe.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --llava_model LLAVA_MODEL_ID --num_images 100
```





