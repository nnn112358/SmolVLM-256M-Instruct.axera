import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from transformers.image_utils import load_image
from modeling_idefics3_export import Idefics3ForConditionalGenerationExport

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = sys.argv[1] if len(sys.argv) >= 2 else "../SmolVLM-256M-Instruct"
# Load images
image = load_image("Statue-of-Liberty-Island-New-York-Bay.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained(checkpoint_dir)


model = Idefics3ForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map=DEVICE
)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"},
        ],
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print("prompt", prompt)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)
print("inputs keys", inputs.keys())
print("input_ids shape", inputs["input_ids"].shape)
print("pixel_values shape", inputs["pixel_values"].shape)
# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
