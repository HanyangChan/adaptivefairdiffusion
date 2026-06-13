import torch
from diffusers import StableDiffusionPipeline
from schedulers import get_fairness_weight

@torch.no_grad()
def generate_with_fairness(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    fairness_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    scheduler_type: str = "linear",
    scheduler_kwargs: dict = None,
    generator: torch.Generator = None,
    eta: float = 0.0,
    adaptive_mode: str = "none",
):
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
        
    device = pipeline.device
    
    # 1. Encode prompts
    # Unconditional
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Text conditioning
    text_input = pipeline.tokenizer(
        [prompt], padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]
    
    # Fairness conditioning
    fair_input = pipeline.tokenizer(
        [fairness_prompt], padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
    )
    fair_embeddings = pipeline.text_encoder(fair_input.input_ids.to(device))[0]
    
    # Concatenate all three for batched inference
    # [uncond, text, fair]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, fair_embeddings])
    
    # 2. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # 3. Prepare latent variables
    latent_shape = (1, pipeline.unet.config.in_channels, 512 // 8, 512 // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=text_embeddings.dtype)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # 4. Denoising loop
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 3)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        # predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance
        noise_pred_uncond, noise_pred_text, noise_pred_fair = noise_pred.chunk(3)
        
        # calculate fairness weight for current step
        fairness_weight = get_fairness_weight(scheduler_type, i, num_inference_steps, **scheduler_kwargs)
        
        d_text = noise_pred_text - noise_pred_uncond
        d_fair = noise_pred_fair - noise_pred_text
        
        if "dynamic_weight" in adaptive_mode:
            # Calculate cosine similarity across all dimensions
            cos_sim = torch.nn.functional.cosine_similarity(d_text.flatten(), d_fair.flatten(), dim=0)
            # Map [-1, 1] to [2, 0]: more conflict means higher weight
            adaptive_factor = 1.0 - cos_sim.item()
            fairness_weight = fairness_weight * adaptive_factor
            
        if "orthogonal" in adaptive_mode:
            # Orthogonalize d_fair with respect to d_text
            dot_fair_text = torch.sum(d_fair * d_text)
            dot_text_text = torch.sum(d_text * d_text)
            projection = (dot_fair_text / (dot_text_text + 1e-8)) * d_text
            d_fair = d_fair - projection
        
        noise_pred = noise_pred_uncond + guidance_scale * d_text + fairness_weight * d_fair
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        
    # 5. Post-processing
    image = pipeline.decode_latents(latents)
    image, has_nsfw_concept = pipeline.run_safety_checker(image, device, text_embeddings.dtype)
    image = pipeline.numpy_to_pil(image)
    
    return image[0]
