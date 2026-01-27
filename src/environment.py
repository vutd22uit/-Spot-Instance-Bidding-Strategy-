# -*- coding: utf-8 -*-
"""
Gymnasium Environment cho Spot Instance Bidding Strategy

Môi trường này mô phỏng việc quản lý cloud instances với 2 loại:
- Spot Instance: Rẻ nhưng có thể bị gián đoạn bất kỳ lúc nào
- On-Demand Instance: Đắt hơn nhưng ổn định

Agent cần học cách tối ưu việc sử dụng các loại instance này
để giảm chi phí trong khi vẫn đảm bảo hoàn thành workload.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from collections import deque

from .data_generator import SpotPriceGenerator, WorkloadGenerator, PriceConfig, WorkloadConfig


@dataclass
class InstanceConfig:
    """Cấu hình cho môi trường Spot Instance"""
    # Giá instance
    on_demand_price: float = 0.10          # Giá on-demand ($/hour)
    spot_price_base: float = 0.03          # Giá spot cơ bản
    spot_price_max: float = 0.08           # Giá spot tối đa

    # Giới hạn instances
    max_instances: int = 10                 # Số instance tối đa
    max_pending_jobs: int = 100             # Số jobs tối đa trong queue

    # Hiệu suất xử lý
    jobs_per_instance_per_step: int = 2     # Jobs mỗi instance xử lý/step

    # Xác suất interruption
    base_interruption_prob: float = 0.05    # Xác suất bị terminate cơ bản
    high_price_interruption_prob: float = 0.15  # Xác suất khi giá cao

    # Thời gian mô phỏng
    episode_length: int = 168               # Số steps/episode (168 = 1 tuần)

    # Checkpointing
    checkpoint_cost: float = 0.01           # Chi phí checkpoint
    checkpoint_recovery_rate: float = 0.8   # Tỷ lệ recovery sau checkpoint


@dataclass
class RewardConfig:
    """Cấu hình reward function"""
    alpha: float = 1.0      # Hệ số phạt chi phí
    beta: float = 0.5       # Hệ số thưởng hoàn thành job
    gamma: float = 2.0      # Hệ số phạt interruption
    delta: float = 1.5      # Hệ số phạt vi phạm SLA

    # Ngưỡng SLA
    sla_max_pending_jobs: int = 50
    sla_max_wait_time: int = 5


@dataclass
class Instance:
    """Đại diện cho một cloud instance"""
    instance_type: str              # "spot" hoặc "on_demand"
    created_at: int                 # Step được tạo
    jobs_processed: int = 0         # Số jobs đã xử lý
    is_checkpointed: bool = False   # Đã checkpoint chưa


class SpotInstanceEnv(gym.Env):
    """
    Gymnasium Environment cho bài toán Spot Instance Bidding.

    State Space (7 features):
        - current_spot_price: Giá spot hiện tại (0 đến max_price)
        - price_moving_avg: Giá trung bình 1 giờ qua
        - hour_of_day: Giờ trong ngày (0-23), normalized to [0,1]
        - day_of_week: Ngày trong tuần (0-6), normalized to [0,1]
        - pending_workload: Số jobs cần xử lý (normalized)
        - active_instances: Số instances đang chạy (normalized)
        - interruption_probability: Xác suất bị terminate (0-1)

    Action Space (5 discrete actions):
        - 0: do_nothing - Không làm gì
        - 1: request_spot_instance - Yêu cầu spot instance mới
        - 2: request_on_demand_instance - Yêu cầu on-demand instance mới
        - 3: terminate_instance - Terminate 1 instance (ưu tiên on-demand trước)
        - 4: checkpoint_and_migrate - Checkpoint spot instances và migrate

    Reward:
        R = -α(cost) + β(jobs_completed) - γ(interruption_penalty) - δ(sla_violation)
    """

    # Metadata cho Gymnasium
    metadata = {'render_modes': ['human', 'ansi']}

    # Định nghĩa các action
    ACTION_DO_NOTHING = 0
    ACTION_REQUEST_SPOT = 1
    ACTION_REQUEST_ON_DEMAND = 2
    ACTION_TERMINATE = 3
    ACTION_CHECKPOINT_MIGRATE = 4

    ACTION_NAMES = {
        0: "do_nothing",
        1: "request_spot",
        2: "request_on_demand",
        3: "terminate",
        4: "checkpoint_migrate"
    }

    def __init__(
        self,
        instance_config: Optional[InstanceConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        workload_pattern: str = "spike",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Khởi tạo môi trường.

        Args:
            instance_config: Cấu hình instance
            reward_config: Cấu hình reward
            workload_pattern: Pattern của workload ("stable", "spike", "random", "periodic")
            seed: Random seed
            render_mode: Chế độ render ("human", "ansi", None)
        """
        super().__init__()

        self.instance_config = instance_config or InstanceConfig()
        self.reward_config = reward_config or RewardConfig()
        self.workload_pattern = workload_pattern
        self.render_mode = render_mode

        # Định nghĩa action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)

        # Định nghĩa observation space (7 continuous features)
        # Tất cả được normalize về khoảng [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Khởi tạo các generator
        price_config = PriceConfig(
            base_price=self.instance_config.spot_price_base,
            max_price=self.instance_config.spot_price_max
        )
        workload_config = WorkloadConfig(
            max_workload=self.instance_config.max_pending_jobs
        )

        self._seed = seed
        self.price_generator = SpotPriceGenerator(config=price_config, seed=seed or 42)
        self.workload_generator = WorkloadGenerator(
            config=workload_config,
            pattern=workload_pattern,
            seed=seed or 42
        )

        # Khởi tạo price history cho moving average
        self.price_history: deque = deque(maxlen=24)  # 24 giờ

        # Trạng thái môi trường
        self.instances: List[Instance] = []
        self.pending_jobs: int = 0
        self.current_step: int = 0
        self.current_hour: int = 0
        self.current_day: int = 0
        self.current_spot_price: float = 0.0

        # Metrics để theo dõi
        self.total_cost: float = 0.0
        self.total_jobs_completed: int = 0
        self.total_interruptions: int = 0
        self.total_sla_violations: int = 0

        # History để phân tích
        self.history: List[Dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset môi trường về trạng thái ban đầu.

        Args:
            seed: Random seed mới (optional)
            options: Options bổ sung (optional)

        Returns:
            Tuple (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self.price_generator = SpotPriceGenerator(seed=seed)
            self.workload_generator = WorkloadGenerator(
                pattern=self.workload_pattern,
                seed=seed
            )

        # Reset generators
        self.price_generator.reset()
        self.workload_generator.reset()

        # Reset trạng thái
        self.instances = []
        self.pending_jobs = 0
        self.current_step = 0
        self.price_history.clear()

        # Reset metrics
        self.total_cost = 0.0
        self.total_jobs_completed = 0
        self.total_interruptions = 0
        self.total_sla_violations = 0
        self.history = []

        # Lấy giá spot ban đầu
        self.current_spot_price, self.current_hour, self.current_day = \
            self.price_generator.step()
        self.price_history.append(self.current_spot_price)

        # Thêm workload ban đầu
        initial_workload = self.workload_generator.step(self.current_hour)
        self.pending_jobs = min(initial_workload, self.instance_config.max_pending_jobs)

        # Tạo observation
        obs = self._get_observation()

        # Thông tin bổ sung
        info = {
            'spot_price': self.current_spot_price,
            'hour': self.current_hour,
            'day': self.current_day,
            'pending_jobs': self.pending_jobs,
            'n_instances': len(self.instances)
        }

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Tạo observation từ trạng thái hiện tại.

        Returns:
            Array 7 features, normalized về [0, 1]
        """
        # 1. Current spot price (normalized)
        price_normalized = self.current_spot_price / self.instance_config.spot_price_max

        # 2. Price moving average (normalized)
        if len(self.price_history) > 0:
            price_avg = np.mean(self.price_history)
        else:
            price_avg = self.current_spot_price
        price_avg_normalized = price_avg / self.instance_config.spot_price_max

        # 3. Hour of day (normalized to [0, 1])
        hour_normalized = self.current_hour / 23.0

        # 4. Day of week (normalized to [0, 1])
        day_normalized = self.current_day / 6.0

        # 5. Pending workload (normalized)
        pending_normalized = self.pending_jobs / self.instance_config.max_pending_jobs

        # 6. Active instances (normalized)
        instances_normalized = len(self.instances) / self.instance_config.max_instances

        # 7. Interruption probability
        interruption_prob = self.price_generator.get_interruption_probability(
            self.current_spot_price
        )

        obs = np.array([
            price_normalized,
            price_avg_normalized,
            hour_normalized,
            day_normalized,
            pending_normalized,
            instances_normalized,
            interruption_prob
        ], dtype=np.float32)

        # Clip về [0, 1] để đảm bảo
        obs = np.clip(obs, 0.0, 1.0)

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện một action và tiến tới step tiếp theo.

        Args:
            action: Action để thực hiện (0-4)

        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Lưu trạng thái trước khi action
        step_info = {
            'step': self.current_step,
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'spot_price': self.current_spot_price,
            'pending_jobs_before': self.pending_jobs,
            'n_instances_before': len(self.instances),
            'n_spot_before': self._count_spot_instances(),
            'n_on_demand_before': self._count_on_demand_instances()
        }

        # Thực hiện action
        action_cost, action_success = self._execute_action(action)

        # Xử lý interruptions cho spot instances
        interruption_penalty, n_interrupted = self._process_interruptions()

        # Xử lý jobs
        jobs_completed = self._process_jobs()

        # Thêm workload mới
        new_workload = self.workload_generator.step(self.current_hour)
        self.pending_jobs = min(
            self.pending_jobs + new_workload,
            self.instance_config.max_pending_jobs
        )

        # Tính chi phí running
        running_cost = self._calculate_running_cost()

        # Tính SLA violation
        sla_violation = self._check_sla_violation()
        if sla_violation > 0:
            self.total_sla_violations += 1

        # Tính reward
        reward = self._calculate_reward(
            action_cost + running_cost,
            jobs_completed,
            interruption_penalty,
            sla_violation
        )

        # Cập nhật metrics
        self.total_cost += action_cost + running_cost
        self.total_jobs_completed += jobs_completed
        self.total_interruptions += n_interrupted

        # Tiến tới step tiếp theo
        self.current_step += 1
        self.current_spot_price, self.current_hour, self.current_day = \
            self.price_generator.step()
        self.price_history.append(self.current_spot_price)

        # Kiểm tra kết thúc episode
        terminated = False  # Không có điều kiện kết thúc sớm
        truncated = self.current_step >= self.instance_config.episode_length

        # Tạo observation mới
        obs = self._get_observation()

        # Cập nhật step info
        step_info.update({
            'action_success': action_success,
            'action_cost': action_cost,
            'running_cost': running_cost,
            'jobs_completed': jobs_completed,
            'n_interrupted': n_interrupted,
            'interruption_penalty': interruption_penalty,
            'sla_violation': sla_violation,
            'reward': reward,
            'pending_jobs_after': self.pending_jobs,
            'n_instances_after': len(self.instances),
            'n_spot_after': self._count_spot_instances(),
            'n_on_demand_after': self._count_on_demand_instances()
        })
        self.history.append(step_info)

        # Info trả về
        info = {
            'step': self.current_step,
            'spot_price': self.current_spot_price,
            'hour': self.current_hour,
            'day': self.current_day,
            'pending_jobs': self.pending_jobs,
            'n_instances': len(self.instances),
            'n_spot': self._count_spot_instances(),
            'n_on_demand': self._count_on_demand_instances(),
            'jobs_completed': jobs_completed,
            'cost': action_cost + running_cost,
            'total_cost': self.total_cost,
            'total_jobs': self.total_jobs_completed,
            'total_interruptions': self.total_interruptions
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """
        Thực hiện action và trả về chi phí và trạng thái thành công.

        Args:
            action: Action cần thực hiện

        Returns:
            Tuple (cost, success)
        """
        cost = 0.0
        success = True

        if action == self.ACTION_DO_NOTHING:
            # Không làm gì
            pass

        elif action == self.ACTION_REQUEST_SPOT:
            # Yêu cầu spot instance mới
            if len(self.instances) < self.instance_config.max_instances:
                new_instance = Instance(
                    instance_type="spot",
                    created_at=self.current_step
                )
                self.instances.append(new_instance)
            else:
                success = False

        elif action == self.ACTION_REQUEST_ON_DEMAND:
            # Yêu cầu on-demand instance mới
            if len(self.instances) < self.instance_config.max_instances:
                new_instance = Instance(
                    instance_type="on_demand",
                    created_at=self.current_step
                )
                self.instances.append(new_instance)
            else:
                success = False

        elif action == self.ACTION_TERMINATE:
            # Terminate instance (ưu tiên on-demand trước để tiết kiệm chi phí)
            if len(self.instances) > 0:
                # Tìm on-demand instance trước
                on_demand_idx = None
                for i, inst in enumerate(self.instances):
                    if inst.instance_type == "on_demand":
                        on_demand_idx = i
                        break

                if on_demand_idx is not None:
                    self.instances.pop(on_demand_idx)
                else:
                    # Nếu không có on-demand, terminate spot instance
                    self.instances.pop(0)
            else:
                success = False

        elif action == self.ACTION_CHECKPOINT_MIGRATE:
            # Checkpoint tất cả spot instances
            spot_count = self._count_spot_instances()
            if spot_count > 0:
                cost = spot_count * self.instance_config.checkpoint_cost
                for inst in self.instances:
                    if inst.instance_type == "spot":
                        inst.is_checkpointed = True
            else:
                success = False

        return cost, success

    def _process_interruptions(self) -> Tuple[float, int]:
        """
        Xử lý interruptions cho spot instances.

        Returns:
            Tuple (total_penalty, n_interrupted)
        """
        total_penalty = 0.0
        n_interrupted = 0
        interruption_prob = self.price_generator.get_interruption_probability(
            self.current_spot_price
        )

        # Lọc ra các spot instances bị interrupt
        surviving_instances = []
        for inst in self.instances:
            if inst.instance_type == "spot":
                if np.random.random() < interruption_prob:
                    # Instance bị interrupt
                    n_interrupted += 1

                    if inst.is_checkpointed:
                        # Có checkpoint - penalty nhẹ hơn
                        total_penalty += 0.5
                        # Tạo instance mới với một phần jobs được recover
                        recovered_jobs = int(inst.jobs_processed *
                                           self.instance_config.checkpoint_recovery_rate)
                        # Reset checkpoint flag
                        inst.is_checkpointed = False
                        inst.jobs_processed = recovered_jobs
                        surviving_instances.append(inst)
                    else:
                        # Không có checkpoint - penalty full
                        total_penalty += 1.0
                else:
                    surviving_instances.append(inst)
            else:
                # On-demand không bị interrupt
                surviving_instances.append(inst)

        self.instances = surviving_instances
        return total_penalty, n_interrupted

    def _process_jobs(self) -> int:
        """
        Xử lý jobs với các instances hiện có.

        Returns:
            Số jobs đã hoàn thành
        """
        if self.pending_jobs == 0 or len(self.instances) == 0:
            return 0

        # Tính tổng capacity
        total_capacity = len(self.instances) * self.instance_config.jobs_per_instance_per_step

        # Số jobs được xử lý
        jobs_completed = min(self.pending_jobs, total_capacity)

        # Cập nhật pending jobs
        self.pending_jobs -= jobs_completed

        # Cập nhật jobs processed cho mỗi instance
        jobs_per_instance = jobs_completed // max(1, len(self.instances))
        for inst in self.instances:
            inst.jobs_processed += jobs_per_instance

        return jobs_completed

    def _calculate_running_cost(self) -> float:
        """
        Tính chi phí running của tất cả instances.

        Returns:
            Tổng chi phí trong step này
        """
        total_cost = 0.0
        for inst in self.instances:
            if inst.instance_type == "spot":
                total_cost += self.current_spot_price
            else:
                total_cost += self.instance_config.on_demand_price
        return total_cost

    def _check_sla_violation(self) -> float:
        """
        Kiểm tra vi phạm SLA.

        Returns:
            Giá trị vi phạm (0 nếu không vi phạm)
        """
        violation = 0.0

        # Vi phạm nếu pending jobs vượt ngưỡng
        if self.pending_jobs > self.reward_config.sla_max_pending_jobs:
            excess_ratio = (self.pending_jobs - self.reward_config.sla_max_pending_jobs) / \
                          self.reward_config.sla_max_pending_jobs
            violation = min(1.0, excess_ratio)

        return violation

    def _calculate_reward(
        self,
        cost: float,
        jobs_completed: int,
        interruption_penalty: float,
        sla_violation: float
    ) -> float:
        """
        Tính reward theo công thức:
        R = -α(cost) + β(jobs_completed) - γ(interruption_penalty) - δ(sla_violation)

        Args:
            cost: Chi phí trong step này
            jobs_completed: Số jobs hoàn thành
            interruption_penalty: Penalty từ interruptions
            sla_violation: Mức độ vi phạm SLA

        Returns:
            Giá trị reward
        """
        reward = (
            -self.reward_config.alpha * cost
            + self.reward_config.beta * jobs_completed
            - self.reward_config.gamma * interruption_penalty
            - self.reward_config.delta * sla_violation
        )
        return reward

    def _count_spot_instances(self) -> int:
        """Đếm số spot instances"""
        return sum(1 for inst in self.instances if inst.instance_type == "spot")

    def _count_on_demand_instances(self) -> int:
        """Đếm số on-demand instances"""
        return sum(1 for inst in self.instances if inst.instance_type == "on_demand")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Lấy các metrics của episode hiện tại.

        Returns:
            Dictionary chứa các metrics
        """
        cost_per_job = self.total_cost / max(1, self.total_jobs_completed)

        return {
            'total_cost': self.total_cost,
            'total_jobs_completed': self.total_jobs_completed,
            'total_interruptions': self.total_interruptions,
            'total_sla_violations': self.total_sla_violations,
            'cost_per_job': cost_per_job,
            'n_steps': self.current_step,
            'final_pending_jobs': self.pending_jobs,
            'final_n_instances': len(self.instances)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử của episode.

        Returns:
            List các step info
        """
        return self.history.copy()

    def render(self) -> Optional[str]:
        """
        Render trạng thái hiện tại của môi trường.

        Returns:
            String mô tả trạng thái (nếu render_mode='ansi')
        """
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Step: {self.current_step} | Hour: {self.current_hour} | Day: {self.current_day}")
        output.append(f"{'='*60}")
        output.append(f"Spot Price: ${self.current_spot_price:.4f}")
        output.append(f"Pending Jobs: {self.pending_jobs}")
        output.append(f"Instances: {len(self.instances)} "
                     f"(Spot: {self._count_spot_instances()}, "
                     f"On-Demand: {self._count_on_demand_instances()})")
        output.append(f"Total Cost: ${self.total_cost:.4f}")
        output.append(f"Total Jobs Completed: {self.total_jobs_completed}")
        output.append(f"Total Interruptions: {self.total_interruptions}")
        output.append(f"{'='*60}")

        render_str = '\n'.join(output)

        if self.render_mode == 'human':
            print(render_str)
        elif self.render_mode == 'ansi':
            return render_str

        return None

    def close(self):
        """Đóng môi trường và giải phóng tài nguyên"""
        pass


# Đăng ký environment với Gymnasium
def register_env():
    """Đăng ký SpotInstanceEnv với Gymnasium registry"""
    try:
        gym.register(
            id='SpotInstance-v0',
            entry_point='src.environment:SpotInstanceEnv',
            max_episode_steps=168,
        )
    except gym.error.Error:
        # Đã được đăng ký rồi
        pass


if __name__ == "__main__":
    # Demo: Test environment
    print("=" * 60)
    print("Demo: SpotInstanceEnv")
    print("=" * 60)

    env = SpotInstanceEnv(render_mode='human')
    obs, info = env.reset(seed=42)

    print(f"\nInitial observation: {obs}")
    print(f"Observation shape: {obs.shape}")

    # Chạy một vài steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {i+1}:")
        print(f"  Action: {env.ACTION_NAMES[action]}")
        print(f"  Reward: {reward:.4f}")
        env.render()

        if terminated or truncated:
            break

    print(f"\n{'='*60}")
    print("Episode Metrics:")
    print("=" * 60)
    metrics = env.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    env.close()
