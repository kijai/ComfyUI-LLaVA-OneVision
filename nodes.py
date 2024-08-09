import torch
from torchvision import transforms
import os
import copy
import hashlib
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .llava.conversation import conv_templates
from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import tokenizer_image_token, process_images
from transformers import set_seed

import warnings
import comfy.model_management as mm
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadLLaVAOneVisionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([ 
                    'lmms-lab/llava-onevision-qwen2-7b-ov',
                    'lmms-lab/llava-onevision-qwen2-0.5b-ov',
                    'lmms-lab/llava-onevision-qwen2-7b-si',
                    'lmms-lab/llava-onevision-qwen2-0.5b-si'
                    
                    ],),
            "device": (["cuda","cpu","mps"],),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),

            },
        }

    RETURN_TYPES = ("LLAVAMODEL",)
    RETURN_NAMES = ("llava_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "LLaVA-OneVision"

    def loadmodel(self, model, device, precision, attention):
        if precision != 'fp32' and device == 'cpu':
            raise ValueError("fp16 and bf16 are not supported on cpu")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]
        print(f"using {attention} for attention")

        model_name = model.split('/')[-1]
        download_path = os.path.join(folder_paths.models_dir, "LLM", "LLaVA-OneVision", model_name)
        
        if not os.path.exists(download_path):
            print(f"Downloading LLaVA-OneVision model to: {download_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            #allow_patterns=[f"*{model}*"],
                            local_dir=download_path,
                            local_dir_use_symlinks=False)

        warnings.filterwarnings("ignore")
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model, 
            None,
            model_name="llava_qwen",
            attn_implementation=attention,
            load_8bit=False, 
            load_4bit=False
            )
        model.eval().to(dtype)

        
        llava_model = {
            'model': model, 
            'tokenizer': tokenizer,
            'image_processor': image_processor,
            'dtype': dtype,
            'device': device
            }

        return (llava_model,)


class LLaVA_OneVision_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llava_model": ("LLAVAMODEL", ),
                "image": ("IMAGE", ),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 4096}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES =("result", )
    FUNCTION = "run"
    CATEGORY = "LLaVA-OneVision"

    def run(self, image, llava_model, prompt, max_tokens, keep_model_loaded):
        offload_device = mm.unet_offload_device()
        model = llava_model["model"]
        tokenizer = llava_model["tokenizer"]
        image_processor = llava_model["image_processor"]
        device = llava_model["device"]
        dtype = llava_model["dtype"]
        
        B, H, W, C = image.shape
        image = image.permute(0, 3, 1, 2)  # Change shape to (B, C, H, W)
        transform = transforms.ToPILImage()
        
        image_pils = [transform(image[i]) for i in range(B)]  # Convert each image to PIL format
        
        image_sizes = [img.size for img in image_pils]  # Get sizes for all images
        image_tensors = process_images(image_pils, image_processor, model.config)  # Process all images
        image_tensors = [_image.to(dtype=dtype, device=device) for _image in image_tensors]  # Move to appropriate device and dtype

        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + prompt

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()       

        input_ids = tokenizer_image_token(
            prompt_question, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
            ).unsqueeze(0).to(device)
        
        model.to(device)
        result = model.generate(
            inputs=input_ids,
            images=image_tensors,
            do_sample=False,
            image_sizes=image_sizes,
            temperature=0,
            max_new_tokens=max_tokens
        )
        if not keep_model_loaded:
            model.to(offload_device)
            mm.soft_empty_cachce()
        text_outputs = tokenizer.batch_decode(result, skip_special_tokens=True)
        print(text_outputs)
        
       
        return (text_outputs,)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadLLaVAOneVisionModel": DownloadAndLoadLLaVAOneVisionModel,
    "LLaVA_OneVision_Run": LLaVA_OneVision_Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLLaVAOneVisionModel": "(Down)Load LLaVA-OneVision Model",
    "LLaVA_OneVision_Run": "LLaVA-OneVision Run",
}
