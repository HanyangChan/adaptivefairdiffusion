import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_fairness_metric(probs):
    p_male = probs.get('a photo of a male', 0)
    p_female = probs.get('a photo of a female', 0)
    return 1.0 - abs(p_male - p_female)

def aggregate_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    aggregated = {}
    for prompt, configs in data.items():
        for config_name, seeds_data in configs.items():
            clip_scores = []
            fairness_scores = []
            for seed_id, metrics in seeds_data.items():
                clip_scores.append(metrics['clip_score'])
                fairness_scores.append(calculate_fairness_metric(metrics['fairness_probs']))
            
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

def plot():
    try:
        data_ext = aggregate_data('results_extended/metrics.json')
    except:
        data_ext = {}
    try:
        data_rev = aggregate_data('results_reverse/metrics.json')
    except:
        data_rev = {}
    try:
        data_opt = aggregate_data('results_option1/metrics.json')
    except:
        data_opt = {}
        
    methods = {
        "Baseline (No Guidance)": [data_opt.get("baseline_none")],
        "Fair Diffusion (Constant)": [data_opt.get("baseline_constant_3.0")],
        "FairGen (Mid 25%)": [data_ext.get("baseline_fairgen")],
        "Ours (Reverse Cosine, Unoptimized)": [data_rev.get("reverse_cosine_5.0"), data_rev.get("reverse_cosine_7.0")],
        "Ours (Reverse Cosine, Optimized)": [data_opt.get("reverse_cosine_5.0"), data_opt.get("reverse_cosine_7.0")]
    }
    
    plt.figure(figsize=(10, 7))
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D', '*']
    
    for (name, points), color, marker in zip(methods.items(), colors, markers):
        valid_points = [p for p in points if p]
        if not valid_points:
            continue
            
        x = [p['fairness'] for p in valid_points]
        y = [p['clip'] for p in valid_points]
        
        plt.scatter(x, y, label=name, color=color, marker=marker, s=150 if name.startswith("Ours") else 100)
        
        if len(x) > 1:
            sorted_idx = np.argsort(x)
            sorted_x = np.array(x)[sorted_idx]
            sorted_y = np.array(y)[sorted_idx]
            plt.plot(sorted_x, sorted_y, color=color, linestyle='--', alpha=0.6)

    plt.title('Quality-Fairness Trade-off: Dynamic Prompting Improvement')
    plt.xlabel('Fairness Score (1.0 = Perfectly Balanced)')
    plt.ylabel('Quality (CLIP Score)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig('tradeoff_plot_optimized.png', dpi=300, bbox_inches='tight')
    print("Plot saved to tradeoff_plot_optimized.png")

if __name__ == '__main__':
    plot()
