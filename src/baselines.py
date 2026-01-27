# -*- coding: utf-8 -*-
"""
Baseline Policies cho Spot Instance Bidding Strategy

Module này chứa các policy baseline để so sánh với RL agent:
- AlwaysOnDemand: Luôn sử dụng on-demand instances (chi phí cao, ổn định)
- AlwaysSpot: Luôn sử dụng spot instances (chi phí thấp, rủi ro cao)
- ThresholdPolicy: Sử dụng spot nếu giá dưới ngưỡng
- RandomPolicy: Chọn action ngẫu nhiên

Mỗi policy implement interface giống nhau để dễ dàng so sánh.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import gymnasium as gym


class BasePolicy(ABC):
    """
    Abstract base class cho tất cả các policy.

    Mỗi policy cần implement method predict() để chọn action
    dựa trên observation hiện tại.
    """

    def __init__(self, name: str = "BasePolicy"):
        """
        Khởi tạo policy.

        Args:
            name: Tên của policy (dùng cho logging/visualization)
        """
        self.name = name

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action dựa trên observation.

        Args:
            observation: State observation từ environment
            deterministic: Có sử dụng policy deterministic không

        Returns:
            Action (0-4)
        """
        pass

    def reset(self) -> None:
        """Reset internal state của policy (nếu có)"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class AlwaysOnDemand(BasePolicy):
    """
    Policy luôn sử dụng on-demand instances.

    Đây là policy an toàn nhất - không có rủi ro bị interrupt
    nhưng chi phí cao nhất.

    Logic:
    - Nếu có pending jobs và chưa đủ instances -> request on-demand
    - Nếu không có pending jobs và có instance -> terminate
    - Còn lại -> do nothing
    """

    # Định nghĩa action constants
    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    def __init__(self, max_instances: int = 5):
        """
        Khởi tạo AlwaysOnDemand policy.

        Args:
            max_instances: Số instances tối đa mà policy sẽ request
        """
        super().__init__(name="AlwaysOnDemand")
        self.max_instances = max_instances

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action.

        Observation indices:
        - 0: current_spot_price (normalized)
        - 1: price_moving_avg (normalized)
        - 2: hour_of_day (normalized)
        - 3: day_of_week (normalized)
        - 4: pending_workload (normalized)
        - 5: active_instances (normalized)
        - 6: interruption_probability

        Args:
            observation: State observation
            deterministic: Không sử dụng (policy luôn deterministic)

        Returns:
            Action
        """
        pending_workload = observation[4]  # Normalized [0, 1]
        active_instances = observation[5]  # Normalized [0, 1]

        # Threshold để quyết định
        workload_threshold = 0.1  # Có jobs cần xử lý
        max_instance_ratio = self.max_instances / 10.0  # Giả sử max_instances env = 10

        if pending_workload > workload_threshold:
            # Có jobs cần xử lý
            if active_instances < max_instance_ratio:
                # Chưa đủ instances -> request on-demand
                return self.ACTION_REQUEST_ON_DEMAND
            else:
                # Đã đủ instances -> do nothing
                return self.ACTION_DO_NOTHING
        else:
            # Không có nhiều jobs
            if active_instances > 0.1:
                # Có instance thừa -> terminate để tiết kiệm
                return self.ACTION_TERMINATE
            else:
                return self.ACTION_DO_NOTHING


class AlwaysSpot(BasePolicy):
    """
    Policy luôn sử dụng spot instances.

    Đây là policy tiết kiệm chi phí nhất nhưng rủi ro cao
    do có thể bị interrupt bất kỳ lúc nào.

    Logic:
    - Nếu có pending jobs và chưa đủ instances -> request spot
    - Nếu interruption probability cao -> checkpoint
    - Nếu không có pending jobs và có instance -> terminate
    """

    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    def __init__(self, max_instances: int = 5, checkpoint_threshold: float = 0.15):
        """
        Khởi tạo AlwaysSpot policy.

        Args:
            max_instances: Số instances tối đa
            checkpoint_threshold: Ngưỡng interruption prob để checkpoint
        """
        super().__init__(name="AlwaysSpot")
        self.max_instances = max_instances
        self.checkpoint_threshold = checkpoint_threshold

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action.

        Args:
            observation: State observation
            deterministic: Không sử dụng

        Returns:
            Action
        """
        pending_workload = observation[4]
        active_instances = observation[5]
        interruption_prob = observation[6]

        workload_threshold = 0.1
        max_instance_ratio = self.max_instances / 10.0

        # Checkpoint nếu có risk cao và có spot instances
        if interruption_prob > self.checkpoint_threshold and active_instances > 0:
            return self.ACTION_CHECKPOINT_MIGRATE

        if pending_workload > workload_threshold:
            if active_instances < max_instance_ratio:
                # Request spot instance
                return self.ACTION_REQUEST_SPOT
            else:
                return self.ACTION_DO_NOTHING
        else:
            if active_instances > 0.1:
                return self.ACTION_TERMINATE
            else:
                return self.ACTION_DO_NOTHING


class ThresholdPolicy(BasePolicy):
    """
    Policy dựa trên ngưỡng giá spot.

    Logic:
    - Nếu giá spot < threshold -> dùng spot
    - Nếu giá spot >= threshold -> dùng on-demand
    - Checkpoint khi interruption prob cao

    Đây là một heuristic đơn giản nhưng hiệu quả trong nhiều trường hợp.
    """

    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    def __init__(
        self,
        spot_price_threshold: float = 0.5,
        max_instances: int = 5,
        checkpoint_threshold: float = 0.15
    ):
        """
        Khởi tạo ThresholdPolicy.

        Args:
            spot_price_threshold: Ngưỡng giá spot (normalized [0,1])
                                  Nếu giá < threshold thì dùng spot
            max_instances: Số instances tối đa
            checkpoint_threshold: Ngưỡng để checkpoint
        """
        super().__init__(name="ThresholdPolicy")
        self.spot_price_threshold = spot_price_threshold
        self.max_instances = max_instances
        self.checkpoint_threshold = checkpoint_threshold

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action dựa trên ngưỡng giá.

        Args:
            observation: State observation
            deterministic: Không sử dụng

        Returns:
            Action
        """
        current_spot_price = observation[0]  # Normalized
        pending_workload = observation[4]
        active_instances = observation[5]
        interruption_prob = observation[6]

        workload_threshold = 0.1
        max_instance_ratio = self.max_instances / 10.0

        # Checkpoint nếu có risk cao
        if interruption_prob > self.checkpoint_threshold and active_instances > 0:
            return self.ACTION_CHECKPOINT_MIGRATE

        if pending_workload > workload_threshold:
            if active_instances < max_instance_ratio:
                # Quyết định dựa trên giá
                if current_spot_price < self.spot_price_threshold:
                    return self.ACTION_REQUEST_SPOT
                else:
                    return self.ACTION_REQUEST_ON_DEMAND
            else:
                return self.ACTION_DO_NOTHING
        else:
            if active_instances > 0.1:
                return self.ACTION_TERMINATE
            else:
                return self.ACTION_DO_NOTHING


class RandomPolicy(BasePolicy):
    """
    Policy chọn action ngẫu nhiên.

    Đây là baseline yếu nhất, dùng để so sánh
    và đảm bảo các policy khác đều tốt hơn random.
    """

    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    def __init__(self, seed: int = 42, action_probs: Optional[List[float]] = None):
        """
        Khởi tạo RandomPolicy.

        Args:
            seed: Random seed
            action_probs: Xác suất cho mỗi action (optional)
                          Nếu None, sử dụng uniform distribution
        """
        super().__init__(name="RandomPolicy")
        self.rng = np.random.default_rng(seed)

        if action_probs is not None:
            assert len(action_probs) == 5, "Phải có 5 xác suất cho 5 actions"
            assert abs(sum(action_probs) - 1.0) < 1e-6, "Tổng xác suất phải bằng 1"
            self.action_probs = action_probs
        else:
            self.action_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Chọn action ngẫu nhiên.

        Args:
            observation: Không sử dụng
            deterministic: Không sử dụng

        Returns:
            Action ngẫu nhiên (0-4)
        """
        return self.rng.choice(5, p=self.action_probs)

    def reset(self) -> None:
        """Reset random generator"""
        pass


class SmartThresholdPolicy(BasePolicy):
    """
    Policy threshold thông minh hơn với nhiều điều kiện.

    Xem xét:
    - Giá spot và trend
    - Workload level
    - Thời gian trong ngày
    - Interruption probability
    """

    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    def __init__(
        self,
        spot_price_threshold: float = 0.5,
        max_instances: int = 5,
        checkpoint_threshold: float = 0.12,
        peak_hours: tuple = (9, 17)  # Giờ cao điểm
    ):
        """
        Khởi tạo SmartThresholdPolicy.

        Args:
            spot_price_threshold: Ngưỡng giá spot
            max_instances: Số instances tối đa
            checkpoint_threshold: Ngưỡng checkpoint
            peak_hours: Tuple (start, end) của giờ cao điểm
        """
        super().__init__(name="SmartThresholdPolicy")
        self.spot_price_threshold = spot_price_threshold
        self.max_instances = max_instances
        self.checkpoint_threshold = checkpoint_threshold
        self.peak_hours = peak_hours

    def _is_peak_hour(self, hour_normalized: float) -> bool:
        """Kiểm tra có phải giờ cao điểm không"""
        hour = int(hour_normalized * 23)
        return self.peak_hours[0] <= hour <= self.peak_hours[1]

    def _is_price_rising(self, current_price: float, avg_price: float) -> bool:
        """Kiểm tra giá có đang tăng không"""
        return current_price > avg_price * 1.1

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Dự đoán action với logic phức tạp hơn.

        Args:
            observation: State observation
            deterministic: Không sử dụng

        Returns:
            Action
        """
        current_spot_price = observation[0]
        price_moving_avg = observation[1]
        hour_normalized = observation[2]
        pending_workload = observation[4]
        active_instances = observation[5]
        interruption_prob = observation[6]

        max_instance_ratio = self.max_instances / 10.0
        is_peak = self._is_peak_hour(hour_normalized)
        price_rising = self._is_price_rising(current_spot_price, price_moving_avg)

        # Checkpoint nếu risk cao
        if interruption_prob > self.checkpoint_threshold and active_instances > 0:
            return self.ACTION_CHECKPOINT_MIGRATE

        # Nếu có workload cao
        if pending_workload > 0.3:  # Workload cao
            if active_instances < max_instance_ratio:
                # Giờ cao điểm hoặc giá đang tăng -> ưu tiên on-demand
                if is_peak or price_rising:
                    if current_spot_price < self.spot_price_threshold * 0.7:
                        # Giá vẫn còn rất thấp -> dùng spot
                        return self.ACTION_REQUEST_SPOT
                    else:
                        return self.ACTION_REQUEST_ON_DEMAND
                else:
                    # Ngoài giờ cao điểm, giá ổn định -> dùng spot
                    if current_spot_price < self.spot_price_threshold:
                        return self.ACTION_REQUEST_SPOT
                    else:
                        return self.ACTION_REQUEST_ON_DEMAND
            return self.ACTION_DO_NOTHING

        elif pending_workload > 0.1:  # Workload trung bình
            if active_instances < max_instance_ratio * 0.5:
                if current_spot_price < self.spot_price_threshold:
                    return self.ACTION_REQUEST_SPOT
                else:
                    return self.ACTION_REQUEST_ON_DEMAND
            return self.ACTION_DO_NOTHING

        else:  # Workload thấp
            if active_instances > 0.1:
                return self.ACTION_TERMINATE
            return self.ACTION_DO_NOTHING


def evaluate_policy(
    policy: BasePolicy,
    env: gym.Env,
    n_episodes: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate một policy trên environment.

    Args:
        policy: Policy cần evaluate
        env: Gymnasium environment
        n_episodes: Số episodes
        seed: Random seed

    Returns:
        Dictionary chứa metrics trung bình
    """
    metrics_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset()

        episode_reward = 0.0
        done = False

        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Lấy metrics từ environment
        ep_metrics = env.get_metrics()
        ep_metrics['episode_reward'] = episode_reward
        metrics_list.append(ep_metrics)

    # Tính trung bình
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[f"mean_{key}"] = np.mean(values)
        avg_metrics[f"std_{key}"] = np.std(values)

    avg_metrics['policy_name'] = policy.name
    avg_metrics['n_episodes'] = n_episodes

    return avg_metrics


def compare_policies(
    policies: List[BasePolicy],
    env: gym.Env,
    n_episodes: int = 10,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    So sánh nhiều policies.

    Args:
        policies: List các policy cần so sánh
        env: Gymnasium environment
        n_episodes: Số episodes mỗi policy
        seed: Random seed

    Returns:
        Dictionary chứa metrics của mỗi policy
    """
    results = {}

    for policy in policies:
        print(f"Evaluating {policy.name}...")
        metrics = evaluate_policy(policy, env, n_episodes, seed)
        results[policy.name] = metrics

    return results


if __name__ == "__main__":
    # Demo: Test các baselines
    from .environment import SpotInstanceEnv

    print("=" * 60)
    print("Demo: Baseline Policies")
    print("=" * 60)

    # Tạo environment
    env = SpotInstanceEnv()

    # Tạo các policies
    policies = [
        AlwaysOnDemand(max_instances=3),
        AlwaysSpot(max_instances=3),
        ThresholdPolicy(spot_price_threshold=0.5, max_instances=3),
        SmartThresholdPolicy(max_instances=3),
        RandomPolicy(seed=42)
    ]

    # So sánh
    results = compare_policies(policies, env, n_episodes=5)

    # In kết quả
    print("\n" + "=" * 60)
    print("Kết quả so sánh:")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Mean Total Cost: ${metrics['mean_total_cost']:.2f}")
        print(f"  Mean Jobs Completed: {metrics['mean_total_jobs_completed']:.0f}")
        print(f"  Mean Cost/Job: ${metrics['mean_cost_per_job']:.4f}")
        print(f"  Mean Interruptions: {metrics['mean_total_interruptions']:.1f}")
        print(f"  Mean Episode Reward: {metrics['mean_episode_reward']:.2f}")
