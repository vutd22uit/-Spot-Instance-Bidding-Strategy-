# -*- coding: utf-8 -*-
"""
Visualization Module cho Spot Instance Bidding Strategy

Module này cung cấp các hàm để visualization:
- Reward curves từ training
- Cost comparison bar charts
- Action distribution
- Price vs Action timeline
- Episode analysis

Sử dụng:
    from src.visualize import plot_training_curves, plot_episode_analysis
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# Cấu hình style chung
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'spot': '#2ca02c',
    'on_demand': '#1f77b4',
    'cost': '#d62728',
    'reward': '#ff7f0e',
    'jobs': '#9467bd',
    'interruption': '#e377c2'
}

ACTION_COLORS = {
    0: '#7f7f7f',  # Do Nothing - Gray
    1: '#2ca02c',  # Request Spot - Green
    2: '#1f77b4',  # Request On-Demand - Blue
    3: '#d62728',  # Terminate - Red
    4: '#ff7f0e'   # Checkpoint - Orange
}

ACTION_NAMES = {
    0: 'Do Nothing',
    1: 'Request Spot',
    2: 'Request On-Demand',
    3: 'Terminate',
    4: 'Checkpoint'
}


def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Vẽ training curves từ TensorBoard logs hoặc CSV.

    Args:
        log_path: Đường dẫn đến log files
        save_path: Đường dẫn lưu plot (optional)
        show: Hiển thị plot

    Returns:
        Figure object
    """
    # Tìm và đọc các file log
    log_dir = Path(log_path)

    # Tìm file monitor.csv nếu có
    monitor_files = list(log_dir.glob("**/monitor.csv"))

    if not monitor_files:
        print(f"No monitor files found in {log_path}")
        return None

    # Đọc và combine data
    all_data = []
    for f in monitor_files:
        try:
            df = pd.read_csv(f, skiprows=1)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No data to plot")
        return None

    data = pd.concat(all_data, ignore_index=True)

    # Tạo figure với subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves - Spot Instance Bidding Strategy', fontsize=14)

    # 1. Episode Reward
    ax1 = axes[0, 0]
    rewards = data['r'].values
    episodes = np.arange(len(rewards))

    # Rolling mean
    window = min(100, len(rewards) // 10)
    if window > 0:
        rolling_mean = pd.Series(rewards).rolling(window=window).mean()
        ax1.plot(episodes, rewards, alpha=0.3, color=COLORS['reward'], label='Episode Reward')
        ax1.plot(episodes, rolling_mean, color=COLORS['reward'], linewidth=2,
                 label=f'Rolling Mean ({window})')
    else:
        ax1.plot(episodes, rewards, color=COLORS['reward'])

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Reward over Training')
    ax1.legend()

    # 2. Episode Length
    ax2 = axes[0, 1]
    lengths = data['l'].values
    ax2.plot(episodes, lengths, alpha=0.5, color='#17becf')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.axhline(y=168, color='r', linestyle='--', label='Expected (168)')
    ax2.legend()

    # 3. Cumulative Reward
    ax3 = axes[1, 0]
    cumsum_rewards = np.cumsum(rewards)
    ax3.plot(episodes, cumsum_rewards, color=COLORS['reward'])
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Reward')

    # 4. Reward Distribution
    ax4 = axes[1, 1]
    ax4.hist(rewards, bins=50, color=COLORS['reward'], alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(rewards), color='r', linestyle='--',
                label=f'Mean: {np.mean(rewards):.2f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_episode_analysis(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Episode Analysis"
) -> plt.Figure:
    """
    Phân tích chi tiết một episode.

    Args:
        history: List các step info từ env.get_history()
        save_path: Đường dẫn lưu plot
        show: Hiển thị plot
        title: Tiêu đề

    Returns:
        Figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(history)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)

    fig.suptitle(f'{title}', fontsize=14)

    # 1. Spot Price Timeline (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['step'], df['spot_price'], color=COLORS['cost'], linewidth=1.5)
    ax1.fill_between(df['step'], df['spot_price'], alpha=0.3, color=COLORS['cost'])
    ax1.set_ylabel('Spot Price ($)')
    ax1.set_title('Giá Spot theo thời gian')

    # Highlight các action
    for action_id in [1, 2]:  # Request spot, Request on-demand
        action_steps = df[df['action'] == action_id]['step']
        action_prices = df[df['action'] == action_id]['spot_price']
        ax1.scatter(action_steps, action_prices, c=ACTION_COLORS[action_id],
                   s=50, zorder=5, label=ACTION_NAMES[action_id])
    ax1.legend(loc='upper right')

    # 2. Actions Timeline
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    scatter_colors = [ACTION_COLORS[a] for a in df['action']]
    ax2.scatter(df['step'], df['action'], c=scatter_colors, s=30, alpha=0.7)
    ax2.set_ylabel('Action')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(list(ACTION_NAMES.values()), fontsize=8)
    ax2.set_title('Actions theo thời gian')

    # 3. Instance Count
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df['step'], df['n_spot_after'], label='Spot', color=COLORS['spot'], linewidth=2)
    ax3.plot(df['step'], df['n_on_demand_after'], label='On-Demand',
             color=COLORS['on_demand'], linewidth=2)
    ax3.plot(df['step'], df['n_instances_after'], label='Total',
             color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Count')
    ax3.set_title('Số Instances')
    ax3.legend()

    # 4. Pending Jobs
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(df['step'], df['pending_jobs_before'], color=COLORS['jobs'], linewidth=1.5)
    ax4.fill_between(df['step'], df['pending_jobs_before'], alpha=0.3, color=COLORS['jobs'])
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Jobs')
    ax4.set_title('Pending Jobs')
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='SLA Threshold')
    ax4.legend()

    # 5. Cumulative Metrics
    ax5 = fig.add_subplot(gs[2, 2])
    cumsum_cost = np.cumsum(df['action_cost'] + df['running_cost'])
    cumsum_jobs = np.cumsum(df['jobs_completed'])

    ax5_twin = ax5.twinx()

    line1, = ax5.plot(df['step'], cumsum_cost, color=COLORS['cost'],
                      linewidth=2, label='Cumulative Cost')
    line2, = ax5_twin.plot(df['step'], cumsum_jobs, color=COLORS['jobs'],
                           linewidth=2, label='Cumulative Jobs')

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Cost ($)', color=COLORS['cost'])
    ax5_twin.set_ylabel('Jobs', color=COLORS['jobs'])
    ax5.set_title('Tích lũy Cost & Jobs')
    ax5.legend(handles=[line1, line2], loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_price_action_timeline(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Vẽ timeline giá spot và actions.

    Args:
        history: Episode history
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    df = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Price vs Action Timeline', fontsize=14)

    # Spot price
    ax1.plot(df['step'], df['spot_price'], color=COLORS['cost'], linewidth=1.5)
    ax1.fill_between(df['step'], df['spot_price'], alpha=0.2, color=COLORS['cost'])

    # Mark interruptions
    interrupted_steps = df[df['n_interrupted'] > 0]['step']
    if len(interrupted_steps) > 0:
        for step in interrupted_steps:
            ax1.axvline(x=step, color=COLORS['interruption'], alpha=0.5, linestyle='--')

    ax1.set_ylabel('Spot Price ($)')
    ax1.set_title('Giá Spot Instance')

    # Actions as colored bars
    for i, row in df.iterrows():
        ax2.bar(row['step'], 1, color=ACTION_COLORS[row['action']], width=1.0)

    ax2.set_xlabel('Step (Hour)')
    ax2.set_ylabel('Action')
    ax2.set_yticks([])

    # Legend
    patches = [mpatches.Patch(color=ACTION_COLORS[i], label=ACTION_NAMES[i])
               for i in range(5)]
    ax2.legend(handles=patches, loc='upper right', ncol=3)
    ax2.set_title('Actions Timeline')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_action_distribution(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Vẽ phân phối actions cho nhiều policies.

    Args:
        results: List kết quả từ evaluate
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    policies = [r['policy_name'] for r in results]
    x = np.arange(len(policies))
    width = 0.6

    bottom = np.zeros(len(policies))

    for action_id in range(5):
        probs = [r['action_distribution'][action_id] for r in results]
        ax.bar(x, probs, width, label=ACTION_NAMES[action_id],
               bottom=bottom, color=ACTION_COLORS[action_id])
        bottom += probs

    ax.set_ylabel('Tỷ lệ Action')
    ax.set_title('Phân phối Actions của các Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_cost_breakdown(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Phân tích chi tiết chi phí.

    Args:
        history: Episode history
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Chi phí chi tiết trong Episode', fontsize=14)

    # 1. Cost per step
    ax1 = axes[0, 0]
    total_cost = df['action_cost'] + df['running_cost']
    ax1.bar(df['step'], df['running_cost'], label='Running Cost',
            color=COLORS['on_demand'], alpha=0.7)
    ax1.bar(df['step'], df['action_cost'], bottom=df['running_cost'],
            label='Action Cost', color=COLORS['cost'], alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Chi phí mỗi Step')
    ax1.legend()

    # 2. Cumulative cost
    ax2 = axes[0, 1]
    cumsum_running = np.cumsum(df['running_cost'])
    cumsum_action = np.cumsum(df['action_cost'])
    ax2.fill_between(df['step'], cumsum_running, label='Running Cost',
                     color=COLORS['on_demand'], alpha=0.5)
    ax2.fill_between(df['step'], cumsum_running, cumsum_running + cumsum_action,
                     label='Action Cost', color=COLORS['cost'], alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Cost ($)')
    ax2.set_title('Tích lũy Chi phí')
    ax2.legend()

    # 3. Cost vs Jobs trade-off
    ax3 = axes[1, 0]
    cumsum_cost = np.cumsum(total_cost)
    cumsum_jobs = np.cumsum(df['jobs_completed'])
    ax3.scatter(cumsum_cost, cumsum_jobs, c=df['step'], cmap='viridis', s=20)
    ax3.set_xlabel('Cumulative Cost ($)')
    ax3.set_ylabel('Cumulative Jobs')
    ax3.set_title('Trade-off: Cost vs Jobs')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3, label='Step')

    # 4. Cost efficiency over time
    ax4 = axes[1, 1]
    efficiency = cumsum_jobs / (cumsum_cost + 0.001)  # Jobs per dollar
    ax4.plot(df['step'], efficiency, color=COLORS['jobs'], linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Jobs per Dollar')
    ax4.set_title('Hiệu quả Chi phí theo thời gian')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_spot_price_patterns(
    n_days: int = 7,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualization patterns giá spot.

    Args:
        n_days: Số ngày để visualize
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    from .data_generator import SpotPriceGenerator

    gen = SpotPriceGenerator(seed=42)
    n_steps = n_days * 24
    prices = gen.generate_price_series(n_steps)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Patterns Giá Spot Instance', fontsize=14)

    # 1. Price over time
    ax1 = axes[0, 0]
    hours = np.arange(n_steps)
    ax1.plot(hours, prices, color=COLORS['cost'], linewidth=1)
    ax1.fill_between(hours, prices, alpha=0.3, color=COLORS['cost'])
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'Giá Spot trong {n_days} ngày')

    # Đánh dấu các ngày
    for day in range(1, n_days):
        ax1.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)

    # 2. Average by hour of day
    ax2 = axes[0, 1]
    prices_by_hour = [prices[i::24] for i in range(24)]
    mean_by_hour = [np.mean(p) for p in prices_by_hour]
    std_by_hour = [np.std(p) for p in prices_by_hour]

    ax2.bar(range(24), mean_by_hour, yerr=std_by_hour, capsize=3,
            color=COLORS['cost'], alpha=0.7)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Price ($)')
    ax2.set_title('Giá trung bình theo Giờ')
    ax2.set_xticks(range(0, 24, 2))

    # 3. Average by day of week
    ax3 = axes[1, 0]
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    prices_reshaped = prices[:n_days * 24].reshape(n_days, 24)

    if n_days >= 7:
        prices_by_day = []
        for d in range(7):
            day_indices = [i for i in range(n_days) if i % 7 == d]
            day_prices = [prices_reshaped[i].mean() for i in day_indices if i < n_days]
            prices_by_day.append(np.mean(day_prices) if day_prices else 0)
    else:
        prices_by_day = [prices_reshaped[d % n_days].mean() for d in range(7)]

    ax3.bar(range(7), prices_by_day, color=COLORS['cost'], alpha=0.7)
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Average Price ($)')
    ax3.set_title('Giá trung bình theo Ngày')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_names)

    # 4. Price distribution
    ax4 = axes[1, 1]
    ax4.hist(prices, bins=50, color=COLORS['cost'], alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(prices), color='r', linestyle='--',
                label=f'Mean: ${np.mean(prices):.4f}')
    ax4.axvline(x=np.median(prices), color='b', linestyle='--',
                label=f'Median: ${np.median(prices):.4f}')
    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Phân phối Giá')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_workload_patterns(
    n_hours: int = 168,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualization các pattern workload.

    Args:
        n_hours: Số giờ để visualize
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    from .data_generator import WorkloadGenerator

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Patterns Workload', fontsize=14)

    patterns = ['stable', 'spike', 'random', 'periodic']
    colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

    for ax, pattern, color in zip(axes.flat, patterns, colors):
        gen = WorkloadGenerator(pattern=pattern, seed=42)
        workloads = gen.generate_workload_series(n_hours)

        ax.plot(range(n_hours), workloads, color=color, linewidth=1)
        ax.fill_between(range(n_hours), workloads, alpha=0.3, color=color)

        ax.set_xlabel('Hour')
        ax.set_ylabel('Jobs')
        ax.set_title(f'Pattern: {pattern.capitalize()}')

        # Statistics
        stats_text = f'Mean: {np.mean(workloads):.1f}\nMax: {max(workloads)}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_dashboard(
    history: List[Dict[str, Any]],
    policy_name: str = "Policy",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Tạo dashboard tổng hợp cho một episode.

    Args:
        history: Episode history
        policy_name: Tên policy
        save_path: Đường dẫn lưu
        show: Hiển thị

    Returns:
        Figure
    """
    df = pd.DataFrame(history)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Dashboard: {policy_name}', fontsize=16, fontweight='bold')

    # 1. Spot Price (row 0, cols 0-2)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(df['step'], df['spot_price'], color=COLORS['cost'], linewidth=1.5)
    ax1.fill_between(df['step'], df['spot_price'], alpha=0.2, color=COLORS['cost'])
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Giá Spot')

    # 2. Summary stats (row 0, col 3)
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_stats.axis('off')

    total_cost = df['action_cost'].sum() + df['running_cost'].sum()
    total_jobs = df['jobs_completed'].sum()
    total_int = df['n_interrupted'].sum()
    cost_per_job = total_cost / max(1, total_jobs)

    stats_text = f"""
    TỔNG KẾT EPISODE

    Tổng Chi phí: ${total_cost:.2f}
    Tổng Jobs: {total_jobs}
    Interruptions: {total_int}
    Cost/Job: ${cost_per_job:.4f}

    Final Instances: {df['n_instances_after'].iloc[-1]}
    - Spot: {df['n_spot_after'].iloc[-1]}
    - On-Demand: {df['n_on_demand_after'].iloc[-1]}
    """
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  verticalalignment='top', fontfamily='monospace', fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # 3. Actions (row 1, cols 0-2)
    ax2 = fig.add_subplot(gs[1, :3])
    for i, row in df.iterrows():
        ax2.bar(row['step'], 1, color=ACTION_COLORS[row['action']], width=1.0)
    ax2.set_ylabel('Action')
    ax2.set_yticks([])
    ax2.set_title('Actions Timeline')

    # 4. Action pie chart (row 1, col 3)
    ax_pie = fig.add_subplot(gs[1, 3])
    action_counts = df['action'].value_counts().sort_index()
    colors_pie = [ACTION_COLORS[i] for i in action_counts.index]
    labels_pie = [ACTION_NAMES[i] for i in action_counts.index]
    ax_pie.pie(action_counts.values, labels=labels_pie, colors=colors_pie,
               autopct='%1.1f%%', startangle=90)
    ax_pie.set_title('Action Distribution')

    # 5. Instances (row 2, cols 0-1)
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.stackplot(df['step'],
                  df['n_spot_after'],
                  df['n_on_demand_after'],
                  labels=['Spot', 'On-Demand'],
                  colors=[COLORS['spot'], COLORS['on_demand']],
                  alpha=0.7)
    ax3.set_ylabel('Instances')
    ax3.set_title('Instance Count')
    ax3.legend(loc='upper right')

    # 6. Pending Jobs (row 2, cols 2-3)
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.plot(df['step'], df['pending_jobs_before'], color=COLORS['jobs'], linewidth=1.5)
    ax4.fill_between(df['step'], df['pending_jobs_before'], alpha=0.3, color=COLORS['jobs'])
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='SLA')
    ax4.set_ylabel('Jobs')
    ax4.set_title('Pending Jobs')
    ax4.legend()

    # 7. Cost & Reward (row 3, cols 0-1)
    ax5 = fig.add_subplot(gs[3, :2])
    cumsum_cost = np.cumsum(df['action_cost'] + df['running_cost'])
    cumsum_reward = np.cumsum(df['reward'])

    ax5_twin = ax5.twinx()
    line1, = ax5.plot(df['step'], cumsum_cost, color=COLORS['cost'],
                      linewidth=2, label='Cost')
    line2, = ax5_twin.plot(df['step'], cumsum_reward, color=COLORS['reward'],
                           linewidth=2, label='Reward')

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Cost ($)', color=COLORS['cost'])
    ax5_twin.set_ylabel('Reward', color=COLORS['reward'])
    ax5.set_title('Tích lũy Cost & Reward')
    ax5.legend(handles=[line1, line2])

    # 8. Jobs Completed (row 3, cols 2-3)
    ax6 = fig.add_subplot(gs[3, 2:])
    cumsum_jobs = np.cumsum(df['jobs_completed'])
    ax6.plot(df['step'], cumsum_jobs, color=COLORS['jobs'], linewidth=2)
    ax6.fill_between(df['step'], cumsum_jobs, alpha=0.3, color=COLORS['jobs'])
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Cumulative Jobs')
    ax6.set_title('Jobs Completed')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Demo: Tạo sample visualizations
    print("=" * 60)
    print("Demo: Visualizations")
    print("=" * 60)

    # Plot patterns
    print("\n1. Spot price patterns:")
    plot_spot_price_patterns(n_days=7, show=True)

    print("\n2. Workload patterns:")
    plot_workload_patterns(n_hours=168, show=True)

    # Demo với episode giả lập
    print("\n3. Episode analysis demo:")
    from .environment import SpotInstanceEnv

    env = SpotInstanceEnv()
    obs, _ = env.reset(seed=42)

    # Chạy một episode với random policy
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    history = env.get_history()
    create_dashboard(history, policy_name="Random Policy", show=True)

    env.close()
