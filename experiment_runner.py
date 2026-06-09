import os
import torch
from diffusers import StableDiffusionPipeline
from evaluate import Evaluator
from pipeline_adaptive_fairness import generate_with_fairness
import json

def run_experiment():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading pipeline from {model_id}...")
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    pipeline = pipeline.to(device)
    
    evaluator = Evaluator(device=device)
    
    prompts = [
        "A portrait of a doctor",
        "A photo of a nurse"
    ]
    fairness_prompt = "diverse ethnicities, equally mixed genders"
    attributes_to_check = ["a photo of a male", "a photo of a female"]
    
    # Ablation configurations
    configs = {
        "baseline_none": {"scheduler_type": "constant", "kwargs": {"weight": 0.0}},
        "baseline_constant_3.0": {"scheduler_type": "constant", "kwargs": {"weight": 3.0}},
        
        "reverse_linear_5.0": {"scheduler_type": "reverse_linear", "kwargs": {"min_w": 0.0, "max_w": 5.0}},
        "reverse_linear_7.0": {"scheduler_type": "reverse_linear", "kwargs": {"min_w": 0.0, "max_w": 7.0}},
        
        "reverse_cosine_5.0": {"scheduler_type": "reverse_cosine", "kwargs": {"min_w": 0.0, "max_w": 5.0}},
        "reverse_cosine_7.0": {"scheduler_type": "reverse_cosine", "kwargs": {"min_w": 0.0, "max_w": 7.0}},
        
        "reverse_exponential_5.0": {"scheduler_type": "reverse_exponential", "kwargs": {"min_w": 0.1, "max_w": 5.0}},
        "reverse_exponential_7.0": {"scheduler_type": "reverse_exponential", "kwargs": {"min_w": 0.1, "max_w": 7.0}},
    }
    
    seeds = [42, 123, 2024]
    
    results = {}
    os.makedirs("results_optimized", exist_ok=True)
    
    for prompt in prompts:
        print(f"\nEvaluating Prompt: '{prompt}'")
        results[prompt] = {}
        
        for config_name, config in configs.items():
            print(f"  Running config: {config_name}")
            results[prompt][config_name] = {}
            
            for seed in seeds:
                print(f"    Seed: {seed}")
                # Use fixed seed for fair comparison
                generator = torch.Generator(device=device).manual_seed(seed)
                
                # Combine base prompt with fairness attributes to preserve original semantics (Quality)
                dynamic_fairness_prompt = f"{prompt}, {fairness_prompt}"
                
                # Generate image
                image = generate_with_fairness(
                    pipeline=pipeline,
                    prompt=prompt,
                    fairness_prompt=dynamic_fairness_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    scheduler_type=config["scheduler_type"],
                    scheduler_kwargs=config["kwargs"],
                    generator=generator
                )
                
                # Save image
                img_path = f"results_optimized/{prompt.replace(' ', '_')}_{config_name}_seed_{seed}.png"
                image.save(img_path)
                
                # Evaluate Quality (CLIP Score)
                clip_score = evaluator.compute_clip_score(image, prompt)
                
                # Evaluate Fairness
                fairness_probs = evaluator.evaluate_fairness(image, attributes_to_check)
                
                results[prompt][config_name][f"seed_{seed}"] = {
                    "clip_score": clip_score,
                    "fairness_probs": fairness_probs,
                    "image_path": img_path
                }
                
                print(f"      CLIP Score: {clip_score:.2f}")
                print(f"      Fairness: {fairness_probs}")

    # Save results
    with open("results_optimized/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nExperiment complete. Results saved to results_optimized/metrics.json")

if __name__ == "__main__":
    run_experiment()
