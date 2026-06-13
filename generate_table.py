import json
import numpy as np

def calculate_fairness_metric(probs):
    p_male = probs.get('a photo of a male', 0)
    p_female = probs.get('a photo of a female', 0)
    return 1.0 - abs(p_male - p_female)

def aggregate_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return {}
        
    aggregated = {}
    for prompt, configs in data.items():
        for config_name, seeds_data in configs.items():
            clip_scores = []
            fairness_scores = []
            
            # Check if it's the old format (no seed_ keys)
            if 'clip_score' in seeds_data:
                clip_scores.append(seeds_data['clip_score'])
                fairness_scores.append(calculate_fairness_metric(seeds_data['fairness_probs']))
            else:
                for seed_id, metrics in seeds_data.items():
                    if isinstance(metrics, dict) and 'clip_score' in metrics:
                        clip_scores.append(metrics['clip_score'])
                        fairness_scores.append(calculate_fairness_metric(metrics['fairness_probs']))
            
            if not clip_scores: continue
            
            avg_clip = np.mean(clip_scores)
            avg_fairness = np.mean(fairness_scores)
            
            if config_name not in aggregated:
                aggregated[config_name] = {'clip': [], 'fairness': []}
            
            aggregated[config_name]['clip'].append(avg_clip)
            aggregated[config_name]['fairness'].append(avg_fairness)
            
    final_stats = {}
    for config_name, metrics in aggregated.items():
        final_stats[config_name] = {
            'clip': np.mean(metrics['clip']),
            'fairness': np.mean(metrics['fairness'])
        }
    return final_stats

def main():
    data_ext = aggregate_data('results_extended/metrics.json')
    data_rev = aggregate_data('results_reverse/metrics.json')
    data_opt = aggregate_data('results_option1/metrics.json')
    
    # We will pick the best performing configs
    methods = [
        ("Stable Diffusion (No Guidance)", data_opt.get("baseline_none")),
        ("FairGen [Mid 25%]", data_ext.get("baseline_fairgen")),
        ("Fair Diffusion [Constant]", data_opt.get("baseline_constant_3.0")),
        ("Ours (Reverse Cosine Unoptimized)", data_rev.get("reverse_cosine_7.0")),
        ("Ours (Reverse Cosine Optimized)", data_opt.get("reverse_cosine_7.0")),
        ("Adaptive (Orthogonal)", data_opt.get("adaptive_orthogonal_rc5.0")),
        ("Adaptive (Dynamic Weight)", data_opt.get("adaptive_dynamic_rc5.0")),
        ("Adaptive (Both)", data_opt.get("adaptive_both_rc5.0"))
    ]
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Fairness $\\uparrow$ & CLIP-score $\\uparrow$ \\\\")
    print("\\midrule")
    
    for name, stats in methods:
        if stats:
            fairness = stats['fairness']
            clip = stats['clip']
            print(f"{name} & {fairness:.3f} & {clip:.2f} \\\\")
        else:
            print(f"{name} & - & - \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Quantitative Evaluation of Quality and Fairness.}")
    print("\\label{tab:quant_eval}")
    print("\\end{table}")

    print("\n\nMarkdown version:")
    print("| Method | Fairness ↑ | CLIP-score ↑ |")
    print("|---|---|---|")
    for name, stats in methods:
        if stats:
            fairness = stats['fairness']
            clip = stats['clip']
            print(f"| {name} | {fairness:.3f} | {clip:.2f} |")

if __name__ == "__main__":
    main()
