# -*- coding: utf-8 -*-
"""
Evaluation Script cho Spot Instance Bidding Strategy

Script này đánh giá và so sánh RL agent với các baselines:
- Tính các metrics: total_cost, jobs_completed, interruptions, cost_per_job
- Vẽ biểu đồ so sánh
- Xuất kết quả ra file CSV

Sử dụng:
    python -m src.evaluate --model models/best_model.zip
    python -m src.evaluate --model models/best_model.zip --n-episodes 50
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

from .environment import SpotInstanceEnv, InstanceConfig, RewardConfig
from .baselines import (
    BasePolicy,
    AlwaysOnDemand,
    AlwaysSpot,
    ThresholdPolicy,
    SmartThresholdPolicy,
    RandomPolicy
)
from .train import load_config, create_env


class DQNPolicyWrapper(BasePolicy):
    """
    Wrapper để DQN model có interface giống BasePolicy.

    Cho phép sử dụng DQN model trong cùng pipeline
    với các baseline policies.
    """

    def __init__(self, model: DQN, name: str = "DQN_Agent"):
        """
        Khởi tạo wrapper.

        Args:
            model: Trained DQN model
            name: Tên để hiển thị
        """
        super().__init__(name=name)
        self.model = model

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action từ model.

        Args:
            observation: State observation
            deterministic: Sử dụng policy deterministic

        Returns:
            Action (0-4)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)


def evaluate_single_episode(
    policy: BasePolicy,
    env: SpotInstanceEnv,
    seed: int = 42,
    render: bool = False
) -> Dict[str, Any]:
    """
    Đánh giá một policy trong một episode.

    Args:
        policy: Policy cần đánh giá
        env: Gymnasium environment
        seed: Random seed
        render: Có render không

    Returns:
        Dictionary chứa metrics và history
    """
    obs, info = env.reset(seed=seed)
    policy.reset()

    episode_reward = 0.0
    done = False
    step_count = 0
    action_counts = {i: 0 for i in range(5)}

    while not done:
        action = policy.predict(obs, deterministic=True)
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        done = terminated or truncated

        if render:
            env.render()

    # Lấy metrics từ environment
    metrics = env.get_metrics()
    metrics['episode_reward'] = episode_reward
    metrics['action_distribution'] = action_counts
    metrics['n_steps'] = step_count

    # Lấy history
    history = env.get_history()

    return {
        'metrics': metrics,
        'history': history
    }


def evaluate_policy_full(
    policy: BasePolicy,
    env: SpotInstanceEnv,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Đánh giá đầy đủ một policy qua nhiều episodes.

    Args:
        policy: Policy cần đánh giá
        env: Gymnasium environment
        n_episodes: Số episodes
        seed: Random seed
        verbose: Có in progress không

    Returns:
        Dictionary chứa metrics statistics
    """
    metrics_list = []

    for ep in range(n_episodes):
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")

        result = evaluate_single_episode(policy, env, seed=seed + ep)
        metrics_list.append(result['metrics'])

    # Tính statistics
    stats = {}

    # Các metrics chính
    metric_keys = [
        'total_cost', 'total_jobs_completed', 'total_interruptions',
        'total_sla_violations', 'cost_per_job', 'episode_reward'
    ]

    for key in metric_keys:
        values = [m[key] for m in metrics_list]
        stats[f'{key}_mean'] = np.mean(values)
        stats[f'{key}_std'] = np.std(values)
        stats[f'{key}_min'] = np.min(values)
        stats[f'{key}_max'] = np.max(values)

    # Action distribution trung bình
    action_totals = {i: 0 for i in range(5)}
    for m in metrics_list:
        for action, count in m['action_distribution'].items():
            action_totals[action] += count

    total_actions = sum(action_totals.values())
    stats['action_distribution'] = {
        k: v / total_actions for k, v in action_totals.items()
    }

    stats['n_episodes'] = n_episodes
    stats['policy_name'] = policy.name

    return stats


def compare_all_policies(
    model_path: Optional[str],
    env: SpotInstanceEnv,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    So sánh tất cả policies (RL agent + baselines).

    Args:
        model_path: Đường dẫn đến trained model
        env: Gymnasium environment
        n_episodes: Số episodes mỗi policy
        seed: Random seed
        verbose: Có in progress không

    Returns:
        DataFrame chứa kết quả so sánh
    """
    policies = []

    # Load RL agent nếu có
    if model_path and os.path.exists(model_path):
        if verbose:
            print(f"Loading model from: {model_path}")
        model = DQN.load(model_path)
        rl_policy = DQNPolicyWrapper(model, name="DQN_Agent")
        policies.append(rl_policy)
    elif verbose:
        print("No model provided, evaluating baselines only")

    # Thêm baselines
    baselines = [
        AlwaysOnDemand(max_instances=5),
        AlwaysSpot(max_instances=5),
        ThresholdPolicy(spot_price_threshold=0.5, max_instances=5),
        SmartThresholdPolicy(max_instances=5),
        RandomPolicy(seed=seed)
    ]
    policies.extend(baselines)

    # Đánh giá từng policy
    results = []

    for policy in policies:
        if verbose:
            print(f"\nEvaluating {policy.name}...")

        stats = evaluate_policy_full(
            policy, env, n_episodes, seed, verbose=verbose
        )
        results.append(stats)

    # Tạo DataFrame
    df_data = []
    for stats in results:
        row = {
            'Policy': stats['policy_name'],
            'Mean Cost': stats['total_cost_mean'],
            'Std Cost': stats['total_cost_std'],
            'Mean Jobs': stats['total_jobs_completed_mean'],
            'Std Jobs': stats['total_jobs_completed_std'],
            'Mean Interruptions': stats['total_interruptions_mean'],
            'Mean Cost/Job': stats['cost_per_job_mean'],
            'Mean Reward': stats['episode_reward_mean'],
            'SLA Violations': stats['total_sla_violations_mean']
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    return df, results


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    In bảng so sánh đẹp.

    Args:
        df: DataFrame kết quả
    """
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    # Format columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print(df.to_string(index=False))

    print("\n" + "-" * 100)

    # Tìm best policy
    best_cost_idx = df['Mean Cost'].idxmin()
    best_jobs_idx = df['Mean Jobs'].idxmax()
    best_reward_idx = df['Mean Reward'].idxmax()
    best_cost_per_job_idx = df['Mean Cost/Job'].idxmin()

    print("\nBest Policies:")
    print(f"  Lowest Cost: {df.loc[best_cost_idx, 'Policy']} "
          f"(${df.loc[best_cost_idx, 'Mean Cost']:.2f})")
    print(f"  Most Jobs: {df.loc[best_jobs_idx, 'Policy']} "
          f"({df.loc[best_jobs_idx, 'Mean Jobs']:.0f})")
    print(f"  Lowest Cost/Job: {df.loc[best_cost_per_job_idx, 'Policy']} "
          f"(${df.loc[best_cost_per_job_idx, 'Mean Cost/Job']:.4f})")
    print(f"  Highest Reward: {df.loc[best_reward_idx, 'Policy']} "
          f"({df.loc[best_reward_idx, 'Mean Reward']:.2f})")


def plot_comparison(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    save_path: str = "./results/plots/"
) -> None:
    """
    Vẽ các biểu đồ so sánh.

    Args:
        df: DataFrame kết quả
        results: List kết quả chi tiết
        save_path: Đường dẫn lưu plots
    """
    os.makedirs(save_path, exist_ok=True)

    # Màu sắc cho các policies
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    # 1. Bar chart - Total Cost
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Policy'], df['Mean Cost'], yerr=df['Std Cost'],
                  color=colors, capsize=5)
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('So sánh Chi phí trung bình')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cost_comparison.png'), dpi=150)
    plt.close()

    # 2. Bar chart - Jobs Completed
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Policy'], df['Mean Jobs'], yerr=df['Std Jobs'],
           color=colors, capsize=5)
    ax.set_ylabel('Jobs Completed')
    ax.set_title('So sánh số Jobs hoàn thành')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'jobs_comparison.png'), dpi=150)
    plt.close()

    # 3. Bar chart - Cost per Job
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Policy'], df['Mean Cost/Job'], color=colors)
    ax.set_ylabel('Cost per Job ($)')
    ax.set_title('So sánh Chi phí trên mỗi Job')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cost_per_job_comparison.png'), dpi=150)
    plt.close()

    # 4. Bar chart - Interruptions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Policy'], df['Mean Interruptions'], color=colors)
    ax.set_ylabel('Number of Interruptions')
    ax.set_title('So sánh số lần bị Interrupt')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'interruptions_comparison.png'), dpi=150)
    plt.close()

    # 5. Action Distribution (Stacked Bar)
    fig, ax = plt.subplots(figsize=(12, 6))
    action_names = ['Do Nothing', 'Request Spot', 'Request On-Demand',
                    'Terminate', 'Checkpoint']
    action_colors = ['#7f7f7f', '#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']

    x = np.arange(len(df))
    width = 0.6

    bottom = np.zeros(len(df))
    for i, action_name in enumerate(action_names):
        action_probs = [r['action_distribution'][i] for r in results]
        ax.bar(x, action_probs, width, label=action_name,
               bottom=bottom, color=action_colors[i])
        bottom += action_probs

    ax.set_ylabel('Action Probability')
    ax.set_title('Phân phối Actions của các Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Policy'], rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'action_distribution.png'), dpi=150)
    plt.close()

    # 6. Combined metrics (Radar chart)
    categories = ['Cost Efficiency', 'Job Completion', 'Stability',
                  'Low Interruptions', 'Reward']

    # Normalize metrics to [0, 1] for radar chart
    def normalize_inverse(values):
        """Normalize sao cho giá trị thấp hơn = tốt hơn"""
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1.0] * len(values)
        return [(max_val - v) / (max_val - min_val) for v in values]

    def normalize_direct(values):
        """Normalize sao cho giá trị cao hơn = tốt hơn"""
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1.0] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    # Tính normalized values
    radar_data = []
    for i in range(len(df)):
        values = [
            normalize_inverse(df['Mean Cost'].tolist())[i],
            normalize_direct(df['Mean Jobs'].tolist())[i],
            normalize_inverse(df['SLA Violations'].tolist())[i],
            normalize_inverse(df['Mean Interruptions'].tolist())[i],
            normalize_direct(df['Mean Reward'].tolist())[i]
        ]
        radar_data.append(values)

    # Plot radar
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (policy_name, data) in enumerate(zip(df['Policy'], radar_data)):
        data_closed = data + data[:1]
        ax.plot(angles, data_closed, 'o-', linewidth=2, label=policy_name)
        ax.fill(angles, data_closed, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('So sánh đa chiều các Policy', size=14, y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'radar_comparison.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {save_path}")


def save_results(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    save_path: str = "./results/"
) -> None:
    """
    Lưu kết quả ra files.

    Args:
        df: DataFrame kết quả
        results: List kết quả chi tiết
        save_path: Đường dẫn lưu
    """
    os.makedirs(save_path, exist_ok=True)

    # Lưu summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_path, f'comparison_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Lưu detailed results
    detailed_path = os.path.join(save_path, f'detailed_results_{timestamp}.yaml')
    with open(detailed_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Detailed results saved to: {detailed_path}")


def evaluate(
    model_path: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    n_episodes: int = 100,
    seed: int = 42,
    workload_pattern: str = "spike",
    save_path: str = "./results/",
    plot: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main evaluation function.

    Args:
        model_path: Đường dẫn trained model
        config_path: Đường dẫn config
        n_episodes: Số episodes
        seed: Random seed
        workload_pattern: Pattern của workload
        save_path: Đường dẫn lưu kết quả
        plot: Có vẽ plots không
        verbose: Có in progress không

    Returns:
        DataFrame kết quả
    """
    print("=" * 60)
    print("Spot Instance Bidding Strategy - Evaluation")
    print("=" * 60)

    # Load config
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = {}

    # Tạo environment
    print(f"\nCreating environment with pattern: {workload_pattern}")
    env = create_env(config, seed=seed, workload_pattern=workload_pattern)

    # Evaluate
    print(f"\nEvaluating policies ({n_episodes} episodes each)...")
    df, results = compare_all_policies(
        model_path, env, n_episodes, seed, verbose
    )

    # In kết quả
    print_comparison_table(df)

    # Lưu kết quả
    save_results(df, results, save_path)

    # Vẽ plots
    if plot:
        plot_path = os.path.join(save_path, 'plots')
        plot_comparison(df, results, plot_path)

    env.close()

    return df


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate DQN agent vs baselines for Spot Instance Bidding"
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Đường dẫn trained model (.zip)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Đường dẫn file config'
    )

    parser.add_argument(
        '--n-episodes', '-n',
        type=int,
        default=100,
        help='Số episodes để evaluate'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--workload-pattern', '-w',
        type=str,
        default='spike',
        choices=['stable', 'spike', 'random', 'periodic'],
        help='Pattern của workload'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='./results/',
        help='Đường dẫn lưu kết quả'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Không vẽ plots'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Không in progress'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.n_episodes,
        seed=args.seed,
        workload_pattern=args.workload_pattern,
        save_path=args.save_path,
        plot=not args.no_plot,
        verbose=not args.quiet
    )
