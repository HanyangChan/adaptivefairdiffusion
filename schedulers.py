import math

def get_fairness_weight(scheduler_type, step, total_steps, **kwargs):
    """
    Returns the fairness guidance weight for a given step and scheduler type.
    step: 0 (most noise) to total_steps - 1 (least noise)
    """
    if scheduler_type == "constant":
        return kwargs.get("weight", 1.0)
        
    elif scheduler_type == "fairgen":
        # FairGen typical: middle 25% steps (e.g., 37% to 62%)
        start_pct = kwargs.get("start_pct", 0.375)
        end_pct = kwargs.get("end_pct", 0.625)
        weight = kwargs.get("weight", 1.0)
        
        current_pct = step / total_steps
        if start_pct <= current_pct <= end_pct:
            return weight
        return 0.0
        
    elif scheduler_type == "linear":
        min_w = kwargs.get("min_w", 0.0)
        max_w = kwargs.get("max_w", 2.0)
        # Step 0 -> min_w, Step total_steps-1 -> max_w
        return min_w + (max_w - min_w) * (step / max(1, total_steps - 1))
        
    elif scheduler_type == "cosine":
        min_w = kwargs.get("min_w", 0.0)
        max_w = kwargs.get("max_w", 2.0)
        # Step 0 -> min_w, Step total_steps-1 -> max_w
        progress = step / max(1, total_steps - 1)
        # Cosine mapping from 0 to 1 over progress 0 to 1
        # cos(pi) = -1, cos(0) = 1. So we want 0.5 * (1 - cos(progress * pi))
        cosine_val = 0.5 * (1 - math.cos(progress * math.pi))
        return min_w + (max_w - min_w) * cosine_val
        
    elif scheduler_type == "exponential":
        min_w = kwargs.get("min_w", 0.1) # shouldn't be exactly 0 for exp
        max_w = kwargs.get("max_w", 2.0)
        progress = step / max(1, total_steps - 1)
        # alpha * exp(beta * progress)
        # at progress=0: alpha = min_w
        # at progress=1: min_w * exp(beta) = max_w => beta = ln(max_w / min_w)
        if min_w <= 0:
            min_w = 1e-4
        beta = math.log(max_w / min_w)
        return min_w * math.exp(beta * progress)
        
    elif scheduler_type == "reverse_linear":
        min_w = kwargs.get("min_w", 0.0)
        max_w = kwargs.get("max_w", 2.0)
        progress = step / max(1, total_steps - 1)
        # Step 0 -> max_w, Step total_steps-1 -> min_w
        return max_w - (max_w - min_w) * progress
        
    elif scheduler_type == "reverse_cosine":
        min_w = kwargs.get("min_w", 0.0)
        max_w = kwargs.get("max_w", 2.0)
        progress = step / max(1, total_steps - 1)
        cosine_val = 0.5 * (1 - math.cos(progress * math.pi))
        return max_w - (max_w - min_w) * cosine_val
        
    elif scheduler_type == "reverse_exponential":
        min_w = kwargs.get("min_w", 0.1) 
        max_w = kwargs.get("max_w", 2.0)
        if min_w <= 0:
            min_w = 1e-4
        progress = step / max(1, total_steps - 1)
        beta = math.log(min_w / max_w)
        return max_w * math.exp(beta * progress)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

if __name__ == '__main__':
    # Test the schedulers
    steps = 10
    print("Linear:", [round(get_fairness_weight("linear", i, steps, min_w=0, max_w=2), 2) for i in range(steps)])
    print("Cosine:", [round(get_fairness_weight("cosine", i, steps, min_w=0, max_w=2), 2) for i in range(steps)])
    print("Exponential:", [round(get_fairness_weight("exponential", i, steps, min_w=0.1, max_w=2), 2) for i in range(steps)])
