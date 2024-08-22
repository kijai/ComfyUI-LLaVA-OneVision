import torch
from torchvision import transforms
import os
import copy
import hashlib
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates
from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import tokenizer_image_token, process_images
from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
from .llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

import hashlib
import warnings
import comfy.model_management as mm
import folder_paths

from comfy.utils import ProgressBar


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
            "precision": (['fp4', 'nf4', 'int8','fp16','bf16','fp32'],
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

        if "16" in precision or "32" in precision:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
            quantized = False
        else:
            dtype = torch.float16
            quantized = True
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
        tokenizer = AutoTokenizer.from_pretrained(download_path)

        if '4' in precision:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=precision
                )
        elif 'int8' in precision:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
                )
        else:
            quantization_config = None
        
        model = LlavaQwenForCausalLM.from_pretrained(
            download_path, 
            low_cpu_mem_usage=True, 
            attn_implementation=attention,
            quantization_config=quantization_config)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()       
        vision_tower.to(device=device, dtype=dtype)
        image_processor = vision_tower.image_processor

        if not quantized:
            model.eval().to(dtype)

        llava_model = {
            'model': model, 
            'tokenizer': tokenizer,
            'image_processor': image_processor,
            'dtype': dtype,
            'device': device,
            'quantized': quantized
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
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES =("result", )
    FUNCTION = "run"
    CATEGORY = "LLaVA-OneVision"

    def run(self, image, llava_model, prompt, max_tokens, keep_model_loaded, temperature, seed):
        offload_device = mm.unet_offload_device()
        model = llava_model["model"]
        tokenizer = llava_model["tokenizer"]
        image_processor = llava_model["image_processor"]
        device = llava_model["device"]
        dtype = llava_model["dtype"]
        
        seed_bytes = str(seed).encode('utf-8')
        hash_object = hashlib.sha256(seed_bytes)
        hashed_seed = int(hash_object.hexdigest(), 16)
        set_seed(hashed_seed % (2**32))
        
        B, H, W, C = image.shape
        image = image.permute(0, 3, 1, 2)  # Change shape to (B, C, H, W)
        transform = transforms.ToPILImage()
        image_pils = [transform(image[i]) for i in range(B)]  # Convert each image to PIL format
        
        image_sizes = [img.size for img in image_pils]  # Get sizes for all images
        
        image_tensors = []
        for image_pil in image_pils:
            processed_image = process_images([image_pil], image_processor, model.config)  # Process individual image
            processed_image = processed_image[0].to(dtype=dtype, device=device)  # Move to appropriate device and dtype
            image_tensors.append(processed_image)

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
        
        if not llava_model["quantized"]:
            model.to(device)
        result = model.generate(
            inputs=input_ids,
            images=image_tensors,
            do_sample=False if temperature == 0.0 else True,
            image_sizes=image_sizes,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        if not keep_model_loaded:
            if not llava_model["quantized"]:
                model.to(offload_device)
                mm.soft_empty_cache()
        text_outputs = tokenizer.batch_decode(result, skip_special_tokens=True)
        print(text_outputs)

       
        return (text_outputs[0],)


class OneVisionCaptionFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llava_model": ("LLAVAMODEL", ),
                "folder_path": ("STRING", ),
                "prompt": ("STRING", {"default": "You are AI captioning tool, you caption images in very elaborate detail without referring to the image as 'the image', the results should be useful for image model training purposes. You focus on the composition, style and action any possible subject is performing. You don't make assumptions or try to tell a story. You also describe the background of the image separately. Caption this image:", "multiline": True} ),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "max_image_size": ("INT", {"default": 1024, "min": 256, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "LLaVA-OneVision"

    def caption(self, folder_path, llava_model, prompt, max_tokens, keep_model_loaded, temperature, seed, max_image_size):
        from PIL import Image
        image_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
        pbar = ProgressBar(len(image_files))

        transform = transforms.ToTensor()
        vision_node = LLaVA_OneVision_Run()
        results_list = []
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize image if it exceeds max_image_size
                if max(img.size) > max_image_size:
                    aspect_ratio = min(max_image_size / img.size[0], max_image_size / img.size[1])
                    new_size = (int(img.size[0] * aspect_ratio), int(img.size[1] * aspect_ratio))
                    img = img.resize(new_size, Image.LANCZOS)

                img_tensor = transform(img)
                img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            except IOError:
                print(f"Cannot open image: {img_path}")
                continue

            result, = vision_node.run(
                llava_model=llava_model,
                image=img_tensor, prompt=prompt, 
                max_tokens=max_tokens, 
                keep_model_loaded=keep_model_loaded, 
                temperature=temperature, 
                seed=seed)

            results_list.append(result)
            
            base_filename = os.path.splitext(img_path)[0]
            with open(f'{base_filename}.txt', 'w') as file:
                file.write(result)
            pbar.update(1)

        return (results_list,)
    
class SaveCaptionToTextFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", ),
                "filename": ("STRING", ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption"

    CATEGORY = "LLaVA-OneVision"

    def caption(self, txt, filename):
        print("SaveCaptionToTextFile: ", txt)
       
        return txt,

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadLLaVAOneVisionModel": DownloadAndLoadLLaVAOneVisionModel,
    "LLaVA_OneVision_Run": LLaVA_OneVision_Run,
    "OneVisionCaptionFolder": OneVisionCaptionFolder,
    "SaveCaptionToTextFile": SaveCaptionToTextFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLLaVAOneVisionModel": "(Down)Load LLaVA-OneVision Model",
    "LLaVA_OneVision_Run": "LLaVA-OneVision Run",
    "OneVisionCaptionFolder": "OneVision Caption Folder",
    "SaveCaptionToTextFile": "SaveCaptionToTextFile",
}
