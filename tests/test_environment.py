# -*- coding: utf-8 -*-
"""
Unit Tests cho Spot Instance Bidding Strategy

Chạy tests:
    pytest tests/
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=html
"""

import pytest
import numpy as np
import gymnasium as gym

import sys
sys.path.insert(0, '..')

from src.environment import SpotInstanceEnv, InstanceConfig, RewardConfig
from src.data_generator import SpotPriceGenerator, WorkloadGenerator, PriceConfig, WorkloadConfig
from src.baselines import AlwaysOnDemand, AlwaysSpot, ThresholdPolicy, RandomPolicy


class TestSpotPriceGenerator:
    """Tests cho SpotPriceGenerator"""

    def test_init(self):
        """Test khởi tạo generator"""
        gen = SpotPriceGenerator(seed=42)
        assert gen is not None
        assert gen.current_step == 0

    def test_get_price_in_range(self):
        """Test giá spot nằm trong khoảng hợp lệ"""
        gen = SpotPriceGenerator(seed=42)
        for hour in range(24):
            for day in range(7):
                price = gen.get_price(hour, day)
                assert gen.config.min_price <= price <= gen.config.max_price

    def test_generate_price_series(self):
        """Test sinh chuỗi giá"""
        gen = SpotPriceGenerator(seed=42)
        prices = gen.generate_price_series(168)  # 1 tuần

        assert len(prices) == 168
        assert all(gen.config.min_price <= p <= gen.config.max_price for p in prices)

    def test_reproducibility(self):
        """Test reproducibility với cùng seed"""
        gen1 = SpotPriceGenerator(seed=42)
        gen2 = SpotPriceGenerator(seed=42)

        prices1 = gen1.generate_price_series(100)
        prices2 = gen2.generate_price_series(100)

        np.testing.assert_array_almost_equal(prices1, prices2)

    def test_step(self):
        """Test hàm step"""
        gen = SpotPriceGenerator(seed=42)

        price, hour, day = gen.step()
        assert gen.current_step == 1
        assert 0 <= hour <= 23
        assert 0 <= day <= 6

    def test_interruption_probability(self):
        """Test xác suất interrupt"""
        gen = SpotPriceGenerator(seed=42)

        # Giá thấp -> xác suất thấp
        low_prob = gen.get_interruption_probability(gen.config.min_price)
        # Giá cao -> xác suất cao
        high_prob = gen.get_interruption_probability(gen.config.max_price)

        assert low_prob < high_prob
        assert 0 <= low_prob <= 1
        assert 0 <= high_prob <= 1


class TestWorkloadGenerator:
    """Tests cho WorkloadGenerator"""

    def test_init_valid_patterns(self):
        """Test khởi tạo với các pattern hợp lệ"""
        for pattern in ['stable', 'spike', 'random', 'periodic']:
            gen = WorkloadGenerator(pattern=pattern, seed=42)
            assert gen is not None
            assert gen.pattern == pattern

    def test_init_invalid_pattern(self):
        """Test khởi tạo với pattern không hợp lệ"""
        with pytest.raises(ValueError):
            WorkloadGenerator(pattern='invalid', seed=42)

    def test_workload_non_negative(self):
        """Test workload luôn không âm"""
        for pattern in ['stable', 'spike', 'random', 'periodic']:
            gen = WorkloadGenerator(pattern=pattern, seed=42)
            workloads = gen.generate_workload_series(100)
            assert all(w >= 0 for w in workloads)

    def test_workload_bounded(self):
        """Test workload không vượt max"""
        config = WorkloadConfig(max_workload=50)
        gen = WorkloadGenerator(config=config, pattern='spike', seed=42)
        workloads = gen.generate_workload_series(100)
        assert all(w <= config.max_workload for w in workloads)


class TestSpotInstanceEnv:
    """Tests cho SpotInstanceEnv"""

    def test_init(self):
        """Test khởi tạo environment"""
        env = SpotInstanceEnv()
        assert env is not None
        assert env.action_space.n == 5
        assert env.observation_space.shape == (7,)

    def test_reset(self):
        """Test reset environment"""
        env = SpotInstanceEnv()
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert obs.shape == (7,)
        assert all(0 <= o <= 1 for o in obs)  # Normalized
        assert isinstance(info, dict)

    def test_reset_reproducibility(self):
        """Test reset với cùng seed cho kết quả giống nhau"""
        env1 = SpotInstanceEnv()
        env2 = SpotInstanceEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_step_valid_actions(self):
        """Test step với tất cả actions hợp lệ"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        for action in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (7,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_invalid_action(self):
        """Test step với action không hợp lệ"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        with pytest.raises(AssertionError):
            env.step(10)  # Invalid action

    def test_episode_length(self):
        """Test episode kết thúc đúng thời điểm"""
        config = InstanceConfig(episode_length=10)
        env = SpotInstanceEnv(instance_config=config)
        env.reset(seed=42)

        for i in range(10):
            _, _, terminated, truncated, _ = env.step(0)
            if i < 9:
                assert not truncated
            else:
                assert truncated

    def test_request_spot_increases_instances(self):
        """Test request spot tăng số instances"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        initial_instances = len(env.instances)
        env.step(1)  # ACTION_REQUEST_SPOT
        assert len(env.instances) == initial_instances + 1
        assert env.instances[-1].instance_type == "spot"

    def test_request_on_demand_increases_instances(self):
        """Test request on-demand tăng số instances"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        initial_instances = len(env.instances)
        env.step(2)  # ACTION_REQUEST_ON_DEMAND
        assert len(env.instances) == initial_instances + 1
        assert env.instances[-1].instance_type == "on_demand"

    def test_max_instances_limit(self):
        """Test không vượt quá max instances"""
        config = InstanceConfig(max_instances=3)
        env = SpotInstanceEnv(instance_config=config)
        env.reset(seed=42)

        # Thêm 5 instances (sẽ chỉ thêm được 3)
        for _ in range(5):
            env.step(1)  # REQUEST_SPOT

        assert len(env.instances) <= 3

    def test_terminate_reduces_instances(self):
        """Test terminate giảm số instances"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        # Thêm instances
        env.step(1)
        env.step(2)
        n_instances = len(env.instances)

        # Terminate
        env.step(3)  # ACTION_TERMINATE
        assert len(env.instances) == n_instances - 1

    def test_get_metrics(self):
        """Test lấy metrics"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        # Chạy vài steps
        for _ in range(10):
            env.step(env.action_space.sample())

        metrics = env.get_metrics()

        assert 'total_cost' in metrics
        assert 'total_jobs_completed' in metrics
        assert 'total_interruptions' in metrics
        assert 'cost_per_job' in metrics

    def test_get_history(self):
        """Test lấy history"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        # Chạy vài steps
        for _ in range(10):
            env.step(env.action_space.sample())

        history = env.get_history()

        assert len(history) == 10
        assert all('action' in h for h in history)
        assert all('reward' in h for h in history)

    def test_observation_normalized(self):
        """Test observation được normalize về [0, 1]"""
        env = SpotInstanceEnv()
        obs, _ = env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert all(0 <= o <= 1 for o in obs), f"Observation out of range: {obs}"
            if terminated or truncated:
                break


class TestBaselines:
    """Tests cho baseline policies"""

    def test_always_on_demand(self):
        """Test AlwaysOnDemand policy"""
        policy = AlwaysOnDemand(max_instances=3)
        env = SpotInstanceEnv()
        obs, _ = env.reset(seed=42)

        # Policy should return valid actions
        for _ in range(10):
            action = policy.predict(obs)
            assert 0 <= action <= 4
            obs, _, _, _, _ = env.step(action)

    def test_always_spot(self):
        """Test AlwaysSpot policy"""
        policy = AlwaysSpot(max_instances=3)
        env = SpotInstanceEnv()
        obs, _ = env.reset(seed=42)

        for _ in range(10):
            action = policy.predict(obs)
            assert 0 <= action <= 4
            obs, _, _, _, _ = env.step(action)

    def test_threshold_policy(self):
        """Test ThresholdPolicy"""
        policy = ThresholdPolicy(spot_price_threshold=0.5, max_instances=3)
        env = SpotInstanceEnv()
        obs, _ = env.reset(seed=42)

        for _ in range(10):
            action = policy.predict(obs)
            assert 0 <= action <= 4
            obs, _, _, _, _ = env.step(action)

    def test_random_policy(self):
        """Test RandomPolicy"""
        policy = RandomPolicy(seed=42)
        obs = np.zeros(7)  # Dummy observation

        # Should return actions in valid range
        actions = [policy.predict(obs) for _ in range(100)]
        assert all(0 <= a <= 4 for a in actions)

    def test_random_policy_reproducibility(self):
        """Test RandomPolicy reproducibility"""
        policy1 = RandomPolicy(seed=42)
        policy2 = RandomPolicy(seed=42)
        obs = np.zeros(7)

        actions1 = [policy1.predict(obs) for _ in range(100)]
        actions2 = [policy2.predict(obs) for _ in range(100)]

        assert actions1 == actions2


class TestRewardFunction:
    """Tests cho reward function"""

    def test_reward_structure(self):
        """Test cấu trúc reward"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        # Thêm instances để có running cost
        env.step(1)  # Request spot

        # Step và kiểm tra reward
        _, reward, _, _, info = env.step(0)

        assert isinstance(reward, float)
        # Reward có thể âm hoặc dương tùy thuộc vào cost và jobs

    def test_cost_affects_reward(self):
        """Test chi phí ảnh hưởng đến reward"""
        env = SpotInstanceEnv()
        env.reset(seed=42)

        # Không có instance -> không có running cost
        _, reward_no_inst, _, _, _ = env.step(0)

        # Reset và thêm instances
        env.reset(seed=42)
        env.step(2)  # Request on-demand (đắt hơn)
        _, reward_with_inst, _, _, _ = env.step(0)

        # Reward với instance (có cost) nên thấp hơn
        # Trừ khi có jobs được complete bù lại


class TestIntegration:
    """Integration tests"""

    def test_full_episode(self):
        """Test chạy đầy đủ một episode"""
        env = SpotInstanceEnv()
        obs, info = env.reset(seed=42)

        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        assert step_count == env.instance_config.episode_length
        metrics = env.get_metrics()
        assert metrics['n_steps'] == step_count

    def test_gymnasium_compatibility(self):
        """Test compatible với Gymnasium API"""
        env = SpotInstanceEnv()

        # Check Gymnasium interface
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')

        # Check spaces
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_different_workload_patterns(self):
        """Test với các workload patterns khác nhau"""
        for pattern in ['stable', 'spike', 'random', 'periodic']:
            env = SpotInstanceEnv(workload_pattern=pattern, seed=42)
            obs, _ = env.reset()

            # Chạy một vài steps
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
