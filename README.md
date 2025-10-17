# Fish Inpainting to Underwater Apartment

Cog deployment for two-stage underwater scene generation and fish inpainting using Stable Diffusion 1.5 with custom LoRAs.

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── models/
│   ├── underwater_lora_best_epoch_43.safetensors    # Stage 1 LoRA
│   └── back_up_this_one_lora_1b_best_epoch_24.safetensors  # Stage 2 LoRA
├── cog.yaml                    # Cog configuration
├── predict.py                  # Main prediction logic
├── .gitattributes              # Git LFS configuration
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Git LFS installed: `git lfs install`
- GitHub account with repository
- Replicate account with API token

### 2. Prepare Repository

```bash
# Initialize Git LFS
git lfs install
git lfs track "*.safetensors"

# Create models directory
mkdir models

# Download LoRA files from Google Drive and place them in models/:
# - underwater_lora_best_epoch_43.safetensors
# - back_up_this_one_lora_1b_best_epoch_24.safetensors

# Add files
git add .gitattributes
git add models/*.safetensors
git add cog.yaml predict.py .github/

# Commit and push
git commit -m "Initial Cog deployment setup"
git push origin main
```

### 3. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add new repository secret:
   - Name: `REPLICATE_API_TOKEN`
   - Value: Your Replicate API token from https://replicate.com/account/api-tokens

### 4. Deploy

Push to `main` branch or trigger workflow manually:

```bash
git push origin main
```

Or use GitHub UI: Actions → Build and Push to Replicate → Run workflow

## Usage

### Stage 1: Generate Underwater Room

```python
import replicate

output = replicate.run(
    "taras-musakovskyi/inpaint-fish-to-underwater-appt",
    input={
        "mode": "generate_underwater_room",
        "underwater_prompt": "underwater living room with light caustics on walls, blue tinted lighting, bubbles floating",
        "seed": 42  # or -1 for random
    }
)
```

### Stage 2: Inpaint Fish

```python
import replicate

output = replicate.run(
    "taras-musakovskyi/inpaint-fish-to-underwater-appt",
    input={
        "mode": "inpaint_fish",
        "image": open("generated_room.png", "rb"),  # from stage 1
        "species": "goldfish",  # or guppy, gold molly, etc.
        "seed": 42
    }
)
```

## Supported Fish Species

- `goldfish` (default)
- `guppy`
- `gold molly`
- `black molly`
- `dalmatian molly`
- `ancistrus catfish`

Each species has optimized:
- Mask dimensions
- Guidance scale
- Mask fill color
- Padding multiplier

## Technical Details

- Base model: Stable Diffusion 1.5
- Upscaler: SD x4 Upscaler
- Hardware: NVIDIA T4 (16GB VRAM)
- Shared components: VAE, text encoder, tokenizer reused between pipelines
- LoRA rank: 32, alpha: 64

## Model Outputs

- **Stage 1**: ~3072×2048 px underwater room scene
- **Stage 2**: Same resolution with inpainted fish

## Troubleshooting

**Build fails with "LoRA files not found":**
- Ensure files are committed with Git LFS: `git lfs ls-files`
- Verify files exist in `models/` directory

**GitHub Actions fails at login:**
- Check `REPLICATE_API_TOKEN` secret is set correctly
- Verify token has push permissions

**Out of memory errors:**
- Model is optimized for T4 16GB
- Ensure `cog.yaml` specifies `gpu: true`