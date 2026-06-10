import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_fairness_metric(probs):
    p_male = probs.get('a photo of a male', 0)
    p_female = probs.get('a photo of a female', 0)
    return 1.0 - abs(p_male - p_female)

def plot():
    try:
        with open('results/metrics.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print("Error loading data:", e)
        return
        
    aggregated = {}
    for prompt, configs in data.items():
        for config_name, metrics in configs.items():
            # In the first experiment, we didn't have seed_ keys, just the metrics directly.
            if 'clip_score' in metrics:
                clip_score = metrics['clip_score']
                fairness_score = calculate_fairness_metric(metrics['fairness_probs'])
            else:
                continue
            
            if config_name not in aggregated:
                aggregated[config_name] = {'clip': [], 'fairness': []}
            
            aggregated[config_name]['clip'].append(clip_score)
            aggregated[config_name]['fairness'].append(fairness_score)
            
    final_stats = {}
    for config_name, metrics in aggregated.items():
        if len(metrics['clip']) > 0:
            final_stats[config_name] = {
                'clip': np.mean(metrics['clip']),
                'fairness': np.mean(metrics['fairness'])
            }
            
    methods = {
        "Baseline (No Guidance)": [final_stats.get("baseline_none")],
        "Fair Diffusion (Constant 1.5)": [final_stats.get("baseline_constant")],
        "FairGen (Mid 25%)": [final_stats.get("baseline_fairgen")],
        "Ours (Forward Linear 3.0)": [final_stats.get("ours_linear")],
        "Ours (Forward Cosine 3.0)": [final_stats.get("ours_cosine")],
        "Ours (Forward Exponential 3.0)": [final_stats.get("ours_exponential")]
    }
    
    plt.figure(figsize=(10, 7))
    colors = ['gray', 'blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', '*', 'v']
    
    for (name, points), color, marker in zip(methods.items(), colors, markers):
        valid_points = [p for p in points if p is not None]
        if not valid_points:
            continue
            
        x = [p['fairness'] for p in valid_points]
        y = [p['clip'] for p in valid_points]
        
        plt.scatter(x, y, label=name, color=color, marker=marker, s=150)

    plt.title('Quality-Fairness Trade-off (Initial Forward Scheduling Experiment)')
    plt.xlabel('Fairness Score (1.0 = Perfectly Balanced)')
    plt.ylabel('Quality (CLIP Score)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig('tradeoff_plot_initial.png', dpi=300, bbox_inches='tight')
    print("Plot saved to tradeoff_plot_initial.png")

if __name__ == '__main__':
    plot()
