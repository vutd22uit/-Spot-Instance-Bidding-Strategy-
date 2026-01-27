# -*- coding: utf-8 -*-
"""
Training Script cho Spot Instance Bidding Strategy

Script này train RL agent (DQN) để học chiến lược tối ưu
sử dụng spot vs on-demand instances.

Sử dụng:
    python -m src.train --config configs/config.yaml
    python -m src.train --total-timesteps 50000 --seed 42
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import SpotInstanceEnv, InstanceConfig, RewardConfig


class MetricsCallback(BaseCallback):
    """
    Custom callback để log các metrics trong quá trình training.

    Logs:
    - Episode rewards
    - Total cost
    - Jobs completed
    - Interruptions
    - Cost per job
    """

    def __init__(self, verbose: int = 0, log_freq: int = 100):
        """
        Khởi tạo MetricsCallback.

        Args:
            verbose: Mức độ verbose
            log_freq: Log mỗi N episodes
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_jobs = []
        self.episode_interruptions = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Được gọi sau mỗi step.

        Returns:
            True để tiếp tục training
        """
        # Kiểm tra xem episode đã kết thúc chưa
        # Lấy info từ môi trường
        infos = self.locals.get('infos', [{}])

        for info in infos:
            if 'episode' in info:
                # Episode đã kết thúc
                self.episode_count += 1
                ep_reward = info['episode']['r']
                self.episode_rewards.append(ep_reward)

                # Log metrics từ environment
                if 'total_cost' in info:
                    self.episode_costs.append(info['total_cost'])
                if 'total_jobs' in info:
                    self.episode_jobs.append(info['total_jobs'])
                if 'total_interruptions' in info:
                    self.episode_interruptions.append(info['total_interruptions'])

                # Log theo frequency
                if self.episode_count % self.log_freq == 0:
                    self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log các metrics hiện tại"""
        if len(self.episode_rewards) == 0:
            return

        # Lấy N episodes gần nhất
        n_recent = min(self.log_freq, len(self.episode_rewards))

        recent_rewards = self.episode_rewards[-n_recent:]
        mean_reward = np.mean(recent_rewards)

        log_str = f"Episode {self.episode_count} | "
        log_str += f"Mean Reward: {mean_reward:.2f}"

        if len(self.episode_costs) >= n_recent:
            mean_cost = np.mean(self.episode_costs[-n_recent:])
            log_str += f" | Mean Cost: ${mean_cost:.2f}"

        if len(self.episode_jobs) >= n_recent:
            mean_jobs = np.mean(self.episode_jobs[-n_recent:])
            log_str += f" | Mean Jobs: {mean_jobs:.0f}"

        if len(self.episode_interruptions) >= n_recent:
            mean_int = np.mean(self.episode_interruptions[-n_recent:])
            log_str += f" | Mean Interruptions: {mean_int:.1f}"

        if self.verbose > 0:
            print(log_str)

        # Log to TensorBoard if available
        if self.logger is not None:
            self.logger.record("custom/mean_episode_reward", mean_reward)
            if len(self.episode_costs) >= n_recent:
                self.logger.record("custom/mean_cost", np.mean(self.episode_costs[-n_recent:]))
            if len(self.episode_jobs) >= n_recent:
                self.logger.record("custom/mean_jobs", np.mean(self.episode_jobs[-n_recent:]))

    def _on_training_end(self) -> None:
        """Được gọi khi training kết thúc"""
        print(f"\n{'='*60}")
        print("Training Summary:")
        print(f"{'='*60}")
        print(f"Total Episodes: {self.episode_count}")

        if len(self.episode_rewards) > 0:
            print(f"Final Mean Reward: {np.mean(self.episode_rewards[-100:]):.2f}")

        if len(self.episode_costs) > 0:
            print(f"Final Mean Cost: ${np.mean(self.episode_costs[-100:]):.2f}")

        if len(self.episode_jobs) > 0:
            print(f"Final Mean Jobs: {np.mean(self.episode_jobs[-100:]):.0f}")


class SaveBestModelCallback(BaseCallback):
    """
    Callback để lưu model tốt nhất dựa trên reward.
    """

    def __init__(
        self,
        save_path: str,
        check_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Khởi tạo SaveBestModelCallback.

        Args:
            save_path: Đường dẫn lưu model
            check_freq: Kiểm tra mỗi N steps
            verbose: Mức độ verbose
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """Kiểm tra và lưu model tốt nhất"""
        # Thu thập episode rewards
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        # Kiểm tra theo frequency
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            # Tính mean reward gần đây
            n_recent = min(100, len(self.episode_rewards))
            mean_reward = np.mean(self.episode_rewards[-n_recent:])

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"New best model! Mean reward: {mean_reward:.2f}")
                    print(f"Model saved to: {self.save_path}")

        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load cấu hình từ file YAML.

    Args:
        config_path: Đường dẫn file config

    Returns:
        Dictionary cấu hình
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_env(
    config: Dict[str, Any],
    seed: Optional[int] = None,
    workload_pattern: str = "spike"
) -> SpotInstanceEnv:
    """
    Tạo environment từ config.

    Args:
        config: Dictionary cấu hình
        seed: Random seed
        workload_pattern: Pattern của workload

    Returns:
        SpotInstanceEnv
    """
    env_config = config.get('environment', {})
    reward_config_dict = config.get('reward', {})

    instance_config = InstanceConfig(
        on_demand_price=env_config.get('on_demand_price', 0.10),
        spot_price_base=env_config.get('spot_price_base', 0.03),
        spot_price_max=env_config.get('spot_price_max', 0.08),
        max_instances=env_config.get('max_instances', 10),
        max_pending_jobs=env_config.get('max_pending_jobs', 100),
        jobs_per_instance_per_step=env_config.get('jobs_per_instance_per_step', 2),
        episode_length=env_config.get('episode_length', 168),
        checkpoint_cost=env_config.get('checkpoint_cost', 0.01),
        checkpoint_recovery_rate=env_config.get('checkpoint_recovery_rate', 0.8)
    )

    reward_config = RewardConfig(
        alpha=reward_config_dict.get('alpha', 1.0),
        beta=reward_config_dict.get('beta', 0.5),
        gamma=reward_config_dict.get('gamma', 2.0),
        delta=reward_config_dict.get('delta', 1.5),
        sla_max_pending_jobs=reward_config_dict.get('sla_max_pending_jobs', 50)
    )

    env = SpotInstanceEnv(
        instance_config=instance_config,
        reward_config=reward_config,
        workload_pattern=workload_pattern,
        seed=seed
    )

    return env


def create_model(
    env: SpotInstanceEnv,
    config: Dict[str, Any],
    tensorboard_log: Optional[str] = None
) -> DQN:
    """
    Tạo DQN model từ config.

    Args:
        env: Gymnasium environment
        config: Dictionary cấu hình
        tensorboard_log: Đường dẫn log TensorBoard

    Returns:
        DQN model
    """
    dqn_config = config.get('dqn', {})

    # Policy kwargs cho network architecture
    policy_kwargs = {
        'net_arch': dqn_config.get('net_arch', [128, 128])
    }

    model = DQN(
        policy=dqn_config.get('policy', 'MlpPolicy'),
        env=env,
        learning_rate=dqn_config.get('learning_rate', 0.0001),
        buffer_size=dqn_config.get('buffer_size', 100000),
        learning_starts=dqn_config.get('learning_starts', 1000),
        batch_size=dqn_config.get('batch_size', 64),
        tau=dqn_config.get('tau', 0.005),
        gamma=dqn_config.get('gamma', 0.99),
        exploration_fraction=dqn_config.get('exploration_fraction', 0.2),
        exploration_initial_eps=dqn_config.get('exploration_initial_eps', 1.0),
        exploration_final_eps=dqn_config.get('exploration_final_eps', 0.05),
        target_update_interval=dqn_config.get('target_update_interval', 500),
        max_grad_norm=dqn_config.get('max_grad_norm', 10),
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=config.get('training', {}).get('verbose', 1),
        seed=config.get('training', {}).get('seed', 42)
    )

    return model


def train(
    config_path: Optional[str] = None,
    total_timesteps: int = 100000,
    seed: int = 42,
    workload_pattern: str = "spike",
    save_path: str = "./models/",
    log_path: str = "./logs/tensorboard/",
    verbose: int = 1
) -> DQN:
    """
    Train DQN agent.

    Args:
        config_path: Đường dẫn file config (optional)
        total_timesteps: Tổng số timesteps
        seed: Random seed
        workload_pattern: Pattern của workload
        save_path: Đường dẫn lưu model
        log_path: Đường dẫn log TensorBoard
        verbose: Mức độ verbose

    Returns:
        Trained DQN model
    """
    print("=" * 60)
    print("Spot Instance Bidding Strategy - Training")
    print("=" * 60)

    # Load config nếu có
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default config")
        config = {}

    # Override config với parameters
    if 'training' not in config:
        config['training'] = {}
    config['training']['total_timesteps'] = total_timesteps
    config['training']['seed'] = seed
    config['training']['verbose'] = verbose

    # Tạo thư mục
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Tạo environment
    print(f"\nCreating environment with pattern: {workload_pattern}")
    env = create_env(config, seed=seed, workload_pattern=workload_pattern)
    env = Monitor(env)

    # Tạo eval environment
    eval_env = create_env(config, seed=seed + 1000, workload_pattern=workload_pattern)
    eval_env = Monitor(eval_env)

    # Tạo model
    print("\nCreating DQN model...")
    model = create_model(env, config, tensorboard_log=log_path)

    # Tạo callbacks
    print("\nSetting up callbacks...")

    # Callback log metrics
    metrics_callback = MetricsCallback(
        verbose=verbose,
        log_freq=config.get('training', {}).get('log_interval', 100)
    )

    # Callback lưu model tốt nhất
    best_model_path = os.path.join(save_path, "best_model")
    save_best_callback = SaveBestModelCallback(
        save_path=best_model_path,
        check_freq=config.get('training', {}).get('eval_freq', 5000),
        verbose=verbose
    )

    # Callback checkpoint định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('training', {}).get('save_freq', 10000),
        save_path=save_path,
        name_prefix="dqn_spot"
    )

    # Callback đánh giá
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=config.get('training', {}).get('eval_freq', 5000),
        n_eval_episodes=config.get('training', {}).get('n_eval_episodes', 10),
        deterministic=True,
        render=False,
        verbose=verbose
    )

    # Kết hợp callbacks
    callbacks = CallbackList([
        metrics_callback,
        save_best_callback,
        checkpoint_callback,
        eval_callback
    ])

    # Training
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs: {log_path}")
    print(f"Model checkpoints: {save_path}")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Lưu model cuối cùng
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Đánh giá cuối cùng
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env,
        n_eval_episodes=20,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for Spot Instance Bidding Strategy"
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Đường dẫn file config YAML'
    )

    parser.add_argument(
        '--total-timesteps', '-t',
        type=int,
        default=100000,
        help='Tổng số timesteps training'
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
        default='./models/',
        help='Đường dẫn lưu model'
    )

    parser.add_argument(
        '--log-path',
        type=str,
        default='./logs/tensorboard/',
        help='Đường dẫn log TensorBoard'
    )

    parser.add_argument(
        '--verbose', '-v',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Mức độ verbose'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = train(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        workload_pattern=args.workload_pattern,
        save_path=args.save_path,
        log_path=args.log_path,
        verbose=args.verbose
    )
