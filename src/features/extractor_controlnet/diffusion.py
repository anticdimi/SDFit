import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from .unet_2d_condition import UNet2DConditionModel

DIFFUSION_MODEL_ID = 'runwayml/stable-diffusion-v1-5'
ckpt = 'diffusion_pytorch_model.fp16.safetensors'
repo = 'runwayml/stable-diffusion-v1-5'


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


def rgb2canny(img):
    input_image = np.asarray(img)
    preprocessor = CannyDetector()
    low_threshold = 100
    high_threshold = 200
    detected_map = preprocessor(input_image, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    return detected_map


def sketch(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred_image = 255 - blurred_image
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    inverted_pencil_sketch_image = 255 - pencil_sketch_image
    rgb = cv2.cvtColor(inverted_pencil_sketch_image, cv2.COLOR_GRAY2RGB) * 10
    return rgb


def rgb2normalmap(normal_map):
    # min_value = np.min(normal_map)
    # max_value = np.max(normal_map)
    # normalized_normal_map = np.where(
    #     normal_map != 0, (normal_map - min_value) / (max_value - min_value), 0
    # )
    if normal_map.max() <= 1:
        normal_map = normal_map * 255
    normal_map_image = (normal_map).astype(np.uint8)
    detected_map = HWC3(normal_map_image)
    return detected_map


def init_controlnet(device: str, control: list[str]) -> StableDiffusionControlNetImg2ImgPipeline:
    unet = UNet2DConditionModel.from_config(DIFFUSION_MODEL_ID, subfolder='unet').to(
        device, torch.float16
    )
    unet.load_state_dict(load_file(hf_hub_download(repo_id=repo, subfolder='unet', filename=ckpt)))

    controlnet = []
    if 'depth' in control:
        controlnet.append(
            ControlNetModel.from_pretrained(
                'lllyasviel/control_v11f1p_sd15_depth',
                torch_dtype=torch.float16,
            )
        )
    if 'canny' in control:
        controlnet.append(
            ControlNetModel.from_pretrained(
                'lllyasviel/control_v11p_sd15_canny',
                torch_dtype=torch.float16,
            )
        )
    if 'normal' in control:
        controlnet.append(
            ControlNetModel.from_pretrained(
                'lllyasviel/control_v11p_sd15_normalbae',
                torch_dtype=torch.float16,
            )
        )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


transform = transforms.ToPILImage()


def process_depth_map(depth, midas_depth=False):
    max_depth = depth.max()
    indices = depth == -1
    if not midas_depth:
        depth = max_depth - depth
    depth[indices] = 0
    max_depth = depth.max()
    depth = depth / max_depth
    depth = transform(depth)
    return depth


def run_diffusion(
    pipe,
    input_image,
    depth_map,
    prompt,
    normal_map_input=None,
    use_latent=False,
    num_images_per_prompt=1,
    return_image=False,
    num_inference_steps=100,
    **kwargs,
):
    control_image = []
    if depth_map is not None:
        control_image.append(
            process_depth_map(depth_map, midas_depth=kwargs.get('midas_depth', False))
        )
    if normal_map_input is not None:
        normal_map = rgb2normalmap(normal_map_input)
        control_image.append(Image.fromarray(normal_map))

    pos_prompt = f'{prompt},photorealistic,real-world,unreal engine,high quality'
    negative_prompt = 'lowres,low quality,blurry,artistic'
    output_type = 'pil'
    if use_latent:
        output_type = 'latent'
    output = pipe(
        pos_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        image=Image.fromarray(input_image),
        control_image=control_image,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=7,
        eta=1,
        output_type=output_type,
        return_image=return_image,
        # generator=generator,
    ).images
    return output


def add_texture_to_render(
    pipe,
    input_image,
    depth_map,
    prompt,
    normal_map_input=None,
    use_latent=False,
    num_images_per_prompt=1,
    return_image=False,
    num_inference_steps=100,
    **kwargs,
):
    return run_diffusion(
        pipe,
        input_image,
        depth_map,
        prompt,
        normal_map_input,
        use_latent=use_latent,
        num_images_per_prompt=num_images_per_prompt,
        return_image=return_image,
        num_inference_steps=num_inference_steps,
        **kwargs,
    )
