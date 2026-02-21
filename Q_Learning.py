"""
Q_Learning.py - Reinforcement learning for lip-sync detection threshold optimization
"""

import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

def read_data_from_txt(filename):
    """Read similarity scores and labels from text file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove trailing comma if exists
            if line.endswith(','):
                line = line[:-1]
            # Parse tuple (sim_val, label)
            sim_val, lbl = eval(line)
            # Ensure correct data types
            sim_val = float(sim_val)
            lbl = int(lbl)
            data.append((sim_val, lbl))
    return data

def compute_metrics(threshold, data):
    """Calculate evaluation metrics for a given threshold"""
    tp, fp, tn, fn = 0, 0, 0, 0
    for sim, lbl in data:
        pred = 1 if sim >= threshold else 0
        if pred == 1 and lbl == 1: tp += 1
        elif pred == 1 and lbl == 0: fp += 1
        elif pred == 0 and lbl == 0: tn += 1
        elif pred == 0 and lbl == 1: fn += 1
    
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(data) if len(data) > 0 else 0
    
    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

def compute_weighted_accuracy(threshold, data):
    """Calculate weighted accuracy for imbalanced datasets"""
    pos_samples = [d for d in data if d[1] == 1]
    neg_samples = [d for d in data if d[1] == 0]
    
    # Calculate weights to balance classes
    weight_pos = len(data) / (2 * len(pos_samples)) if len(pos_samples) > 0 else 0
    weight_neg = len(data) / (2 * len(neg_samples)) if len(neg_samples) > 0 else 0
    
    # Count correct predictions for each class
    correct_pos = sum(1 for sim, lbl in pos_samples if (sim >= threshold) == (lbl == 1))
    correct_neg = sum(1 for sim, lbl in neg_samples if (sim < threshold) == (lbl == 0))
    
    # Calculate weighted accuracy
    weighted_acc = 0
    if (weight_pos * len(pos_samples) + weight_neg * len(neg_samples)) > 0:
        weighted_acc = (weight_pos * correct_pos + weight_neg * correct_neg) / (weight_pos * len(pos_samples) + weight_neg * len(neg_samples))
    
    return weighted_acc

def get_threshold_range(data, percentile_min=5, percentile_max=95):
    """Determine appropriate threshold range based on data distribution"""
    sim_values = [sim for sim, _ in data]
    if not sim_values:
        return 0.75, 0.95  # Default values if no data
    
    min_threshold = max(0.0, np.percentile(sim_values, percentile_min))
    max_threshold = min(1.0, np.percentile(sim_values, percentile_max))
    
    # Ensure reasonable range
    if max_threshold - min_threshold < 0.1:
        center = (max_threshold + min_threshold) / 2
        min_threshold = max(0.0, center - 0.05)
        max_threshold = min(1.0, center + 0.05)
    
    return min_threshold, max_threshold

def q_learning_threshold(data, 
                         n_episodes=5000, 
                         alpha=0.1, 
                         gamma=0.95, 
                         epsilon=0.1,
                         n_thresholds=21,
                         use_weighted_accuracy=True,
                         min_threshold=None,
                         max_threshold=None):
    """
    Q-Learning algorithm for threshold optimization
    Returns optimal threshold and Q-table
    """
    # Determine threshold range if not specified
    if min_threshold is None or max_threshold is None:
        min_threshold, max_threshold = get_threshold_range(data)
    
    print(f"Using threshold range: [{min_threshold:.3f}, {max_threshold:.3f}]")
    
    # Discretize threshold space
    threshold_space = np.linspace(min_threshold, max_threshold, n_thresholds)
    
    # Initialize Q-table (3 actions: -1, 0, +1)
    Q = np.zeros((n_thresholds, 3), dtype=float)

    # Start Q-Learning training
    for episode in range(n_episodes):
        # Adaptive learning rate and exploration rate
        effective_alpha = alpha * (1 - episode / n_episodes * 0.9)
        effective_epsilon = epsilon * (1 - episode / n_episodes * 0.9)
        
        # Random initial state index
        state_idx = random.randint(0, n_thresholds-1)
        
        done = False
        while not done:
            # Possible actions: -1 (decrease), 0 (stay), +1 (increase)
            possible_moves = [-1, 0, +1]
            
            # ε-greedy action selection
            if random.random() < effective_epsilon:
                # Explore: random action
                action_idx = random.randint(0, 2)
            else:
                # Exploit: best action
                action_idx = np.argmax(Q[state_idx, :])
            
            action = possible_moves[action_idx]
            
            # Calculate next state index
            next_state_idx = state_idx + action
            # Boundary check
            if next_state_idx < 0:
                next_state_idx = 0
            elif next_state_idx > n_thresholds-1:
                next_state_idx = n_thresholds-1
            
            # Get threshold value for next state
            threshold_val = threshold_space[next_state_idx]
            
            # Calculate reward (accuracy or weighted accuracy)
            if use_weighted_accuracy:
                reward = compute_weighted_accuracy(threshold_val, data)
            else:
                metrics = compute_metrics(threshold_val, data)
                reward = metrics["f1"]  # Use F1 score as reward
            
            # Q-Learning update
            best_next_Q = np.max(Q[next_state_idx, :])
            old_q = Q[state_idx, action_idx]
            new_q = old_q + effective_alpha * (reward + gamma * best_next_Q - old_q)
            Q[state_idx, action_idx] = new_q
            
            # Decide whether to end episode
            if action == 0:
                done = True
            else:
                state_idx = next_state_idx
                if random.random() < 0.1 + (episode / n_episodes * 0.4):  # Increasing chance to end episode
                    done = True
        
        # Print progress periodically
        if (episode + 1) % (n_episodes // 10) == 0:
            best_state = np.argmax(np.max(Q, axis=1))
            best_thr = threshold_space[best_state]
            print(f"Episode {episode+1}/{n_episodes}: Current best threshold = {best_thr:.3f}, epsilon = {effective_epsilon:.3f}")

    # Select highest Q-value threshold
    best_state = np.argmax(np.max(Q, axis=1))
    best_threshold = threshold_space[best_state]

    return best_threshold, Q

def find_best_threshold_sklearn(data, method='f1'):
    """
    Find optimal threshold using scikit-learn methods
    Supports ROC, PR curve, or direct F1 optimization
    """
    if not data:
        return 0.85  # Default if no data
    
    # Extract scores and labels
    y_true = np.array([lbl for _, lbl in data])
    scores = np.array([sim for sim, _ in data])
    
    if method == 'roc':
        # ROC curve method - maximize TPR-FPR
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        best_idx = np.argmax(tpr - fpr)
        if best_idx < len(thresholds):
            return thresholds[best_idx]
    
    elif method == 'pr':
        # PR curve method - maximize F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # Add 1.0 threshold
        thresholds = np.append(thresholds, 1.0)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            return thresholds[best_idx]
    
    elif method == 'f1':
        # Direct F1 optimization
        thresholds = np.linspace(0, 1, 100)
        best_threshold = thresholds[0]
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    # Default fallback
    return 0.85

def cross_validate_threshold(data, n_splits=5, n_episodes=5000, alpha=0.1, gamma=0.95, epsilon=0.1):
    """
    Cross-validate threshold settings across multiple folds
    Returns summary of different threshold methods
    """
    if len(data) < n_splits:
        print(f"Warning: Not enough data for {n_splits} folds. Using single fold.")
        n_splits = 1
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Extract indices and labels for KFold
    X = np.array(range(len(data)))
    y = np.array([lbl for _, lbl in data])
    
    q_thresholds = []
    roc_thresholds = []
    pr_thresholds = []
    f1_thresholds = []
    
    # Get global threshold range to use consistently
    min_threshold, max_threshold = get_threshold_range(data)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nProcessing fold {fold+1}/{n_splits}")
        
        # Split data into train and validation sets
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        print(f"Training set: {len(train_data)} samples, Validation set: {len(val_data)} samples")
        
        # Q-Learning
        q_threshold, _ = q_learning_threshold(
            train_data, 
            n_episodes=n_episodes, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon,
            min_threshold=min_threshold,
            max_threshold=max_threshold
        )
        
        # Scikit-learn methods
        roc_threshold = find_best_threshold_sklearn(train_data, method='roc')
        pr_threshold = find_best_threshold_sklearn(train_data, method='pr')
        f1_threshold = find_best_threshold_sklearn(train_data, method='f1')
        
        # Store thresholds
        q_thresholds.append(q_threshold)
        roc_thresholds.append(roc_threshold)
        pr_thresholds.append(pr_threshold)
        f1_thresholds.append(f1_threshold)
        
        # Evaluate on validation set
        q_metrics = compute_metrics(q_threshold, val_data)
        roc_metrics = compute_metrics(roc_threshold, val_data)
        pr_metrics = compute_metrics(pr_threshold, val_data)
        f1_metrics = compute_metrics(f1_threshold, val_data)
        
        print(f"Q-Learning threshold: {q_threshold:.3f}, F1: {q_metrics['f1']:.3f}")
        print(f"ROC curve threshold: {roc_threshold:.3f}, F1: {roc_metrics['f1']:.3f}")
        print(f"PR curve threshold: {pr_threshold:.3f}, F1: {pr_metrics['f1']:.3f}")
        print(f"F1 direct threshold: {f1_threshold:.3f}, F1: {f1_metrics['f1']:.3f}")
    
    # Calculate average thresholds
    avg_q_threshold = np.mean(q_thresholds)
    avg_roc_threshold = np.mean(roc_thresholds)
    avg_pr_threshold = np.mean(pr_thresholds)
    avg_f1_threshold = np.mean(f1_thresholds)
    
    # Final evaluation on complete dataset
    q_metrics = compute_metrics(avg_q_threshold, data)
    roc_metrics = compute_metrics(avg_roc_threshold, data)
    pr_metrics = compute_metrics(avg_pr_threshold, data)
    f1_metrics = compute_metrics(avg_f1_threshold, data)
    
    summary = {
        'q_learning': {'threshold': avg_q_threshold, 'metrics': q_metrics},
        'roc_curve': {'threshold': avg_roc_threshold, 'metrics': roc_metrics},
        'pr_curve': {'threshold': avg_pr_threshold, 'metrics': pr_metrics},
        'f1_direct': {'threshold': avg_f1_threshold, 'metrics': f1_metrics}
    }
    
    return summary

def print_summary(summary):
    """Print threshold optimization results in readable format"""
    print("\n" + "="*80)
    print("Threshold Optimization Summary")
    print("="*80)
    
    methods = ['q_learning', 'roc_curve', 'pr_curve', 'f1_direct']
    method_names = {
        'q_learning': 'Q-Learning',
        'roc_curve': 'ROC Curve',
        'pr_curve': 'PR Curve',
        'f1_direct': 'F1 Direct'
    }
    
    print("\nAverage Thresholds and Overall Performance:")
    print("-"*80)
    print(f"{'Method':<15} {'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-"*80)
    
    for method in methods:
        threshold = summary[method]['threshold']
        metrics = summary[method]['metrics']
        print(f"{method_names[method]:<15} {threshold:.3f}       {metrics['accuracy']:.3f}       {metrics['precision']:.3f}       {metrics['recall']:.3f}       {metrics['f1']:.3f}")
    
    print("\nRecommendation:")
    # Find method with highest F1 score
    best_method = max(methods, key=lambda m: summary[m]['metrics']['f1'])
    best_threshold = summary[best_method]['threshold']
    best_f1 = summary[best_method]['metrics']['f1']
    
    print(f"Best method: {method_names[best_method]} threshold {best_threshold:.3f} (F1: {best_f1:.3f})")
    print("="*80)

if __name__ == "__main__":
    # Read data from text file
    data = read_data_from_txt("Training_data.txt")
    
    if not data:
        print("No data loaded. Exiting.")
        exit(1)
    
    # Data statistics
    positive_count = sum(1 for _, lbl in data if lbl == 1)
    negative_count = sum(1 for _, lbl in data if lbl == 0)
    print(f"Data stats: {len(data)} total samples")
    print(f"Positive (lip-sync) samples: {positive_count} ({positive_count/len(data)*100:.1f}%)")
    print(f"Negative (real singing) samples: {negative_count} ({negative_count/len(data)*100:.1f}%)")
    
    # Cross-validation
    print("\nStarting cross-validation...")
    summary = cross_validate_threshold(
        data,
        n_splits=5,
        n_episodes=3000,
        alpha=0.08,
        gamma=0.9,
        epsilon=0.15
    )
    
    # Print results
    print_summary(summary)
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    best_threshold, Q_table = q_learning_threshold(
        data,
        n_episodes=5000,
        alpha=0.08,
        gamma=0.9,
        epsilon=0.15
    )
    
    print(f"\nFinal recommended threshold: {best_threshold:.3f}")
    print(f"Performance on full dataset: {compute_metrics(best_threshold, data)}")