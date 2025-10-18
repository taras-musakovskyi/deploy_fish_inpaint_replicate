import os
import json
import random
import shutil
import torch
from typing import Literal
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    EulerAncestralDiscreteScheduler
)
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFilter

# Species-specific configurations
SPECIES_GUIDANCE = {
    "guppy": 9.5,
    "gold molly": 6.5,
    "black molly": 6.5,
    "dalmatian molly": 6.5,
    "ancistrus catfish": 6.5,
    "goldfish": 6.5,
}

SPECIES_MASK_FILL = {
    "guppy": 225,
    "gold molly": 225,
    "black molly": 225,
    "dalmatian molly": 225,
    "ancistrus catfish": 240,
    "goldfish": 240,
}

SPECIES_PAD_MULTIPLIER = {
    "guppy": 0.2,
    "goldfish": 0.2,
    "gold molly": 0.2,
    "black molly": 0.2,
    "dalmatian molly": 0.2,
    "ancistrus catfish": 0.2,
}


class Predictor(BasePredictor):
    def setup(self):
        """Load shared models and components"""
        print("🔧 Starting model setup...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "runwayml/stable-diffusion-v1-5"
        
        # Paths to baked-in LoRAs
        self.underwater_lora_path = "/src/models/underwater_lora_best_epoch_43.safetensors"
        self.fish_lora_path = "/src/models/back_up_this_one_lora_1b_best_epoch_24.safetensors"
        
        print(f"📦 Loading base SD 1.5 pipeline on {self.device}...")
        # Load base generation pipeline
        self.pipe_generate = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipe_generate.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_generate.scheduler.config
        )
        
        print("📦 Loading inpainting pipeline (shares components)...")
        # Load inpainting pipeline (will share base components)
        self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            vae=self.pipe_generate.vae,  # Share VAE
            text_encoder=self.pipe_generate.text_encoder,  # Share text encoder
            tokenizer=self.pipe_generate.tokenizer  # Share tokenizer
        )
        self.pipe_inpaint.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_inpaint.scheduler.config
        )
        
        print("📦 Loading upscaler pipeline...")
        # Load upscaler
        self.pipe_upscale = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe_upscale.enable_attention_slicing()
        self.pipe_upscale.enable_vae_tiling()
        
        # Move to device
        self.pipe_generate = self.pipe_generate.to(self.device)
        self.pipe_inpaint = self.pipe_inpaint.to(self.device)
        self.pipe_upscale = self.pipe_upscale.to(self.device)
        
        print("✅ Model setup complete!")
    
    def _apply_lora(self, pipeline, lora_path: str, adapter_name: str):
        """Apply LoRA to pipeline UNet"""
        print(f"🔌 Applying LoRA: {adapter_name}...")
        
        adapter_dir = f"/tmp/lora_{adapter_name}"
        shutil.rmtree(adapter_dir, ignore_errors=True)
        os.makedirs(adapter_dir, exist_ok=True)
        shutil.copy(lora_path, os.path.join(adapter_dir, "adapter_model.safetensors"))
        
        adapter_config = {
            "base_model_name_or_path": self.model_name,
            "peft_type": "LORA",
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "lora_dropout": 0.1
        }
        
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            adapter_dir,
            adapter_name=adapter_name
        )
        
        print(f"✅ LoRA applied: {adapter_name}")
        return pipeline
    
    def predict(
        self,
        mode: Literal["generate_underwater_room", "inpaint_fish"] = Input(
            description="Operation mode: generate base room or inpaint fish"
        ),
        # Stage 1 parameters
        underwater_prompt: str = Input(
            description="Prompt for underwater room generation (stage 1 only)",
            default="underwater living room with light caustics on walls, blue tinted lighting, bubbles floating"
        ),
        seed: int = Input(
            description="Random seed for generation. -1 for random",
            default=-1
        ),
        # Stage 2 parameters
        image: Path = Input(
            description="Base image to inpaint into (stage 2 only)",
            default=None
        ),
        mask_image: Path = Input(
            description="Mask image (grayscale PNG, white=inpaint area) (stage 2 only)",
            default=None
        ),
        species: str = Input(
            description="Fish species to inpaint",
            default="goldfish",
            choices=["goldfish", "guppy", "gold molly", "black molly", "dalmatian molly", "ancistrus catfish"]
        ),
    ) -> Path:
        """Main prediction function"""
        
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        print(f"🎲 Using seed: {seed}")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        if mode == "generate_underwater_room":
            return self._generate_underwater_room(underwater_prompt, generator)
        elif mode == "inpaint_fish":
            if image is None:
                raise ValueError("Image is required for inpaint_fish mode")
            if mask_image is None:
                raise ValueError("Mask image is required for inpaint_fish mode")
            return self._inpaint_fish(image, mask_image, species, generator)
    
    def _generate_underwater_room(self, prompt: str, generator):
        """Stage 1 + 1.1: Generate and upscale underwater room"""
        print("🌊 Starting underwater room generation...")
        
        # Apply underwater LoRA
        self.pipe_generate = self._apply_lora(
            self.pipe_generate,
            self.underwater_lora_path,
            "underwater_apartment"
        )
        
        negative_prompt = "blurry, low quality, distorted, cartoon, multiple rooms, cropped"
        
        print(f"🎨 Generating base image with prompt: '{prompt}'")
        image = self.pipe_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        
        # Crop 2% borders (data augmentation artifact fix)
        print("✂️ Cropping borders...")
        w, h = image.size
        crop_px_w = int(w * 0.02)
        crop_px_h = int(h * 0.02)
        image = image.crop((crop_px_w, crop_px_h, w - crop_px_w, h - crop_px_h))
        
        # Resize to 768x512 for upscaling
        print("📐 Resizing to 768x512 for upscaling...")
        image = image.resize((768, 512), Image.LANCZOS)
        
        # Upscale 4x
        print("🔍 Upscaling 4x with SD Upscaler...")
        image = self.pipe_upscale(
            prompt=prompt,
            image=image,
            num_inference_steps=12
        ).images[0]
        
        output_path = "/tmp/generated_room.png"
        image.save(output_path)
        print(f"✅ Underwater room generated and saved: {image.size}")
        
        return Path(output_path)
    
    def _inpaint_fish(self, base_image_path: Path, mask_image_path: Path, species: str, generator):
        """Stage 2: Inpaint fish into base image"""
        print(f"🐟 Starting fish inpainting for species: {species}")
        
        # Apply fish LoRA
        self.pipe_inpaint = self._apply_lora(
            self.pipe_inpaint,
            self.fish_lora_path,
            "fish_lora"
        )
        
        # Load base image
        print("📷 Loading base image...")
        init_image = Image.open(str(base_image_path)).convert("RGB")
        w, h = init_image.size
        print(f"📐 Base image size: {w}x{h}")
        
        # Load mask from UI (binary: 0=keep, 255=inpaint)
        print("🎭 Loading mask image...")
        mask_image = Image.open(str(mask_image_path)).convert("L")
        
        # Get species-specific configs
        guidance_scale = SPECIES_GUIDANCE[species]
        mask_fill = SPECIES_MASK_FILL[species]
        pad_multiplier = SPECIES_PAD_MULTIPLIER[species]
        
        # Adjust mask fill value based on species (convert binary to species-specific)
        print(f"🎨 Adjusting mask fill to species-specific value: {mask_fill}")
        mask_image = mask_image.point(lambda p: mask_fill if p > 128 else 0)
        
        # Find mask bounding box to determine crop region
        bbox = mask_image.getbbox()
        if bbox is None:
            raise ValueError("Mask is empty - no inpaint region detected")
        
        mask_x0, mask_y0, mask_x1, mask_y1 = bbox
        mask_w = mask_x1 - mask_x0
        mask_h = mask_y1 - mask_y0
        cx = (mask_x0 + mask_x1) // 2
        cy = (mask_y0 + mask_y1) // 2
        
        print(f"📍 Mask center: ({cx}, {cy}), size: {mask_w}x{mask_h}")
        
        # Calculate pad before applying blur to the mask
        pad = int(mask_w * pad_multiplier)

        # Apply blur to mask
        blur_radius = int(mask_w * 0.08)
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Determine crop box with padding
        x0 = max(0, cx - mask_w // 2 - pad)
        y0 = max(0, cy - mask_h // 2 - pad)
        x1 = min(w, cx + mask_w // 2 + pad)
        y1 = min(h, cy + mask_h // 2 + pad)
        
        crop_box = (x0, y0, x1, y1)
        print(f"✂️ Crop box: {crop_box}, padding multiplier: {pad_multiplier}")
        
        region = init_image.crop(crop_box)
        region_mask = mask_image.crop(crop_box)
        
        # Resize to 512x512 for inpainting
        print("📐 Resizing region to 512x512 for inpainting...")
        region_small = region.resize((512, 512), Image.LANCZOS)
        mask_small = region_mask.resize((512, 512), Image.LANCZOS)
        
        SPECIES_PROMPTS_INPAINT = {
            "gold molly": "a complete single gold molly fish with head, body and tail visible, full fish from head to tail, highly detailed, swimming in aquarium, in clear water, sharp focus",
            "guppy": "a complete single guppy fish with head, body and tail visible, full fish from head to tail, highly detailed, colorful tale, in clear water, sharp focus",
            "goldfish": "a complete single goldfish with head, body and tail visible, full fish from head to tail, highly detailed, in clear water, sharp focus",
            "ancistrus catfish": "a complete single ancistrus catfish with head, body and tail visible, full fish from head to tail, highly detailed, on substrate, sharp focus",
            "black molly": "a complete single black molly fish with head, body and tail visible, full fish from head to tail, highly detailed, sharp focus",
            "dalmatian molly": "a complete single dalmatian molly fish with head, body and tail visible, full fish from head to tail, highly detailed, swimming in aquarium, sharp focus",
        }

        prompt = SPECIES_PROMPTS_INPAINT.get(species, "a stunning colorful fish swimming gracefully in a crystal clear aquarium with perfect lighting")
        
        negative_prompt = (
            "multiple fish, duplicate, extra eyes, multiple eyes, double eyes, four eyes, "
            "deformed eyes, extra fins, multiple fins, duplicate fins, mutated, distorted, "
            "disfigured, malformed, duplicate body parts, motion blur, blurry, smooth, "
            "headless, no head, missing head, cropped head, decapitated, partial fish, cut off "
            "cropped, partial, cut off, incomplete, half fish, head only, tail only, body cut, edge of frame"
            "plastic, metallic, aquarium glass, ribbed glass"
        )
        
        print(f"🎨 Inpainting with prompt: '{prompt}'")
        print(f"⚙️ Guidance scale: {guidance_scale}")
        
        out_small = self.pipe_inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=region_small,
            mask_image=mask_small,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Resize back and paste into original
        print("🔄 Resizing inpainted region back and compositing...")
        out_large = out_small.resize(region.size, Image.LANCZOS)
        final_image = init_image.copy()
        final_image.paste(out_large, crop_box)
        
        output_path = "/tmp/inpainted_fish.png"
        final_image.save(output_path)
        print(f"✅ Fish inpainted successfully!")
        
        return Path(output_path)