import torch
from splitfuser.pipelines import SplitSDXLPipeline
from splitfuser.utils import SplitConfig

split_ratio_list = [0.25, 0.75]

split_config = SplitConfig(height=1024, width=1024, warmup_steps=4, split_ratio_list=split_ratio_list, do_classifier_free_guidance=True)
pipeline = SplitSDXLPipeline.from_pretrained(
    split_config=split_config,
    # pretrained_model_name_or_path="/root/autodl-fs/model/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    pretrained_model_name_or_path="/home/jaydenlee/model/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    variant="fp16",
    use_safetensors=True,
)

pipeline.set_progress_bar_config(disable=split_config.rank != 0)
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if split_config.rank == 0:
    image.save("astronaut.png")
