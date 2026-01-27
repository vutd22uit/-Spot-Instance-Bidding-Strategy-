# -*- coding: utf-8 -*-
"""
Data Generator cho Spot Instance Bidding Strategy

Module này chứa các class để sinh dữ liệu giá spot và workload
với các pattern thực tế như:
- Giá spot thay đổi theo giờ trong ngày
- Giá spot thay đổi theo ngày trong tuần
- Workload với các pattern: stable, spike, random, periodic
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PriceConfig:
    """Cấu hình cho việc sinh giá spot"""
    base_price: float = 0.03        # Giá cơ bản ($/hour)
    max_price: float = 0.08         # Giá tối đa
    min_price: float = 0.01         # Giá tối thiểu
    daily_amplitude: float = 0.02   # Biên độ dao động theo ngày
    weekly_amplitude: float = 0.01  # Biên độ dao động theo tuần
    noise_std: float = 0.005        # Độ lệch chuẩn nhiễu


@dataclass
class WorkloadConfig:
    """Cấu hình cho việc sinh workload"""
    base_workload: int = 10         # Số jobs cơ bản mỗi step
    spike_multiplier: float = 5.0   # Hệ số nhân khi có spike
    spike_probability: float = 0.1  # Xác suất xảy ra spike
    max_workload: int = 100         # Workload tối đa


class SpotPriceGenerator:
    """
    Class sinh dữ liệu giá spot instance với các pattern thực tế.

    Giá spot thay đổi theo:
    - Giờ trong ngày: Cao hơn vào giờ làm việc (9-17h)
    - Ngày trong tuần: Thấp hơn vào cuối tuần
    - Nhiễu ngẫu nhiên
    """

    def __init__(self, config: Optional[PriceConfig] = None, seed: int = 42):
        """
        Khởi tạo SpotPriceGenerator.

        Args:
            config: Cấu hình giá spot, nếu None sử dụng giá trị mặc định
            seed: Random seed để reproducibility
        """
        self.config = config or PriceConfig()
        self.rng = np.random.default_rng(seed)
        self.current_step = 0

    def reset(self) -> None:
        """Reset generator về trạng thái ban đầu"""
        self.current_step = 0

    def _get_hour_factor(self, hour: int) -> float:
        """
        Tính hệ số giá theo giờ trong ngày.

        Giá cao hơn vào giờ làm việc (9-17h), đỉnh điểm lúc 13h.
        Giá thấp hơn vào ban đêm (0-6h).

        Args:
            hour: Giờ trong ngày (0-23)

        Returns:
            Hệ số nhân giá (0.5 đến 1.5)
        """
        # Sử dụng hàm sin để mô phỏng pattern theo giờ
        # Đỉnh lúc 13h (peak business hours)
        peak_hour = 13
        factor = np.sin(np.pi * (hour - peak_hour + 12) / 24)
        # Scale về khoảng 0.7 đến 1.3
        return 1.0 + 0.3 * factor

    def _get_day_factor(self, day: int) -> float:
        """
        Tính hệ số giá theo ngày trong tuần.

        Giá thấp hơn vào cuối tuần (thứ 7, CN).
        Giá cao nhất vào giữa tuần (thứ 3, 4).

        Args:
            day: Ngày trong tuần (0=Monday, 6=Sunday)

        Returns:
            Hệ số nhân giá (0.8 đến 1.2)
        """
        # Thứ 3 và thứ 4 có giá cao nhất
        # Cuối tuần có giá thấp nhất
        if day in [5, 6]:  # Thứ 7, CN
            return 0.8
        elif day in [2, 3]:  # Thứ 3, 4
            return 1.2
        else:
            return 1.0

    def get_price(self, hour: int, day: int) -> float:
        """
        Lấy giá spot tại thời điểm cụ thể.

        Args:
            hour: Giờ trong ngày (0-23)
            day: Ngày trong tuần (0-6)

        Returns:
            Giá spot ($/hour)
        """
        # Giá cơ bản
        price = self.config.base_price

        # Áp dụng hệ số theo giờ
        hour_factor = self._get_hour_factor(hour)
        price += self.config.daily_amplitude * (hour_factor - 1)

        # Áp dụng hệ số theo ngày
        day_factor = self._get_day_factor(day)
        price += self.config.weekly_amplitude * (day_factor - 1)

        # Thêm nhiễu ngẫu nhiên
        noise = self.rng.normal(0, self.config.noise_std)
        price += noise

        # Giới hạn giá trong khoảng cho phép
        price = np.clip(price, self.config.min_price, self.config.max_price)

        return price

    def generate_price_series(self, n_steps: int, start_hour: int = 0,
                              start_day: int = 0) -> np.ndarray:
        """
        Sinh chuỗi giá spot cho n_steps.

        Args:
            n_steps: Số steps cần sinh
            start_hour: Giờ bắt đầu
            start_day: Ngày bắt đầu

        Returns:
            Array giá spot cho mỗi step
        """
        prices = np.zeros(n_steps)

        for i in range(n_steps):
            # Tính giờ và ngày hiện tại
            total_hours = start_hour + i
            current_hour = total_hours % 24
            current_day = (start_day + total_hours // 24) % 7

            prices[i] = self.get_price(current_hour, current_day)

        return prices

    def step(self) -> Tuple[float, int, int]:
        """
        Tiến một step và trả về giá, giờ, ngày.

        Returns:
            Tuple (price, hour, day)
        """
        hour = self.current_step % 24
        day = (self.current_step // 24) % 7
        price = self.get_price(hour, day)
        self.current_step += 1
        return price, hour, day

    def get_interruption_probability(self, price: float) -> float:
        """
        Tính xác suất bị interrupt dựa trên giá.

        Giá càng cao thì xác suất interrupt càng cao.

        Args:
            price: Giá spot hiện tại

        Returns:
            Xác suất interrupt (0 đến 1)
        """
        # Normalize giá về khoảng 0-1
        price_normalized = (price - self.config.min_price) / \
                          (self.config.max_price - self.config.min_price)

        # Xác suất interrupt tăng theo giá
        # Base prob: 0.02 khi giá thấp nhất
        # Max prob: 0.20 khi giá cao nhất
        base_prob = 0.02
        max_prob = 0.20

        prob = base_prob + (max_prob - base_prob) * price_normalized
        return prob


class WorkloadGenerator:
    """
    Class sinh workload với các pattern khác nhau.

    Hỗ trợ các pattern:
    - stable: Workload ổn định
    - spike: Có đột biến ngẫu nhiên
    - random: Hoàn toàn ngẫu nhiên
    - periodic: Theo chu kỳ (cao vào ban ngày)
    """

    def __init__(self, config: Optional[WorkloadConfig] = None,
                 pattern: str = "stable", seed: int = 42):
        """
        Khởi tạo WorkloadGenerator.

        Args:
            config: Cấu hình workload
            pattern: Loại pattern ("stable", "spike", "random", "periodic")
            seed: Random seed
        """
        self.config = config or WorkloadConfig()
        self.pattern = pattern
        self.rng = np.random.default_rng(seed)
        self.current_step = 0

        # Validate pattern
        valid_patterns = ["stable", "spike", "random", "periodic"]
        if pattern not in valid_patterns:
            raise ValueError(f"Pattern phải là một trong {valid_patterns}")

    def reset(self) -> None:
        """Reset generator về trạng thái ban đầu"""
        self.current_step = 0

    def _get_stable_workload(self) -> int:
        """
        Sinh workload ổn định với nhiễu nhỏ.

        Returns:
            Số jobs cần xử lý
        """
        noise = self.rng.integers(-2, 3)
        workload = self.config.base_workload + noise
        return max(0, workload)

    def _get_spike_workload(self) -> int:
        """
        Sinh workload với khả năng có spike.

        Returns:
            Số jobs cần xử lý
        """
        base = self._get_stable_workload()

        # Có xác suất xảy ra spike
        if self.rng.random() < self.config.spike_probability:
            spike = int(base * self.config.spike_multiplier)
            return min(spike, self.config.max_workload)

        return base

    def _get_random_workload(self) -> int:
        """
        Sinh workload hoàn toàn ngẫu nhiên.

        Returns:
            Số jobs cần xử lý
        """
        return self.rng.integers(0, self.config.max_workload // 2)

    def _get_periodic_workload(self, hour: int) -> int:
        """
        Sinh workload theo chu kỳ ngày.

        Workload cao vào giờ làm việc, thấp vào ban đêm.

        Args:
            hour: Giờ trong ngày (0-23)

        Returns:
            Số jobs cần xử lý
        """
        # Đỉnh workload lúc 10-14h
        if 10 <= hour <= 14:
            multiplier = 2.0
        elif 8 <= hour <= 18:
            multiplier = 1.5
        elif 6 <= hour <= 22:
            multiplier = 1.0
        else:
            multiplier = 0.3

        workload = int(self.config.base_workload * multiplier)
        noise = self.rng.integers(-2, 3)

        return max(0, min(workload + noise, self.config.max_workload))

    def get_workload(self, hour: int = 0) -> int:
        """
        Lấy workload theo pattern đã cấu hình.

        Args:
            hour: Giờ trong ngày (dùng cho periodic pattern)

        Returns:
            Số jobs cần xử lý
        """
        if self.pattern == "stable":
            return self._get_stable_workload()
        elif self.pattern == "spike":
            return self._get_spike_workload()
        elif self.pattern == "random":
            return self._get_random_workload()
        elif self.pattern == "periodic":
            return self._get_periodic_workload(hour)
        else:
            return self._get_stable_workload()

    def step(self, hour: int = 0) -> int:
        """
        Tiến một step và trả về workload.

        Args:
            hour: Giờ trong ngày

        Returns:
            Số jobs mới đến
        """
        workload = self.get_workload(hour)
        self.current_step += 1
        return workload

    def generate_workload_series(self, n_steps: int,
                                  start_hour: int = 0) -> np.ndarray:
        """
        Sinh chuỗi workload cho n_steps.

        Args:
            n_steps: Số steps cần sinh
            start_hour: Giờ bắt đầu

        Returns:
            Array workload cho mỗi step
        """
        workloads = np.zeros(n_steps, dtype=int)

        for i in range(n_steps):
            hour = (start_hour + i) % 24
            workloads[i] = self.get_workload(hour)

        return workloads


def generate_sample_data(n_days: int = 7, seed: int = 42) -> pd.DataFrame:
    """
    Sinh dữ liệu mẫu bao gồm giá spot và workload.

    Args:
        n_days: Số ngày dữ liệu
        seed: Random seed

    Returns:
        DataFrame chứa timestamp, spot_price, workload
    """
    n_steps = n_days * 24  # Mỗi step là 1 giờ

    # Khởi tạo generators
    price_gen = SpotPriceGenerator(seed=seed)
    workload_gen = WorkloadGenerator(pattern="spike", seed=seed)

    # Sinh dữ liệu
    data = []
    for i in range(n_steps):
        price, hour, day = price_gen.step()
        workload = workload_gen.step(hour)

        data.append({
            'step': i,
            'hour': hour,
            'day_of_week': day,
            'spot_price': round(price, 4),
            'workload': workload
        })

    return pd.DataFrame(data)


def save_sample_data(filepath: str, n_days: int = 7, seed: int = 42) -> None:
    """
    Sinh và lưu dữ liệu mẫu vào file CSV.

    Args:
        filepath: Đường dẫn file output
        n_days: Số ngày dữ liệu
        seed: Random seed
    """
    df = generate_sample_data(n_days, seed)
    df.to_csv(filepath, index=False)
    print(f"Đã lưu dữ liệu mẫu vào {filepath}")
    print(f"Số dòng: {len(df)}")
    print(f"Các cột: {list(df.columns)}")


if __name__ == "__main__":
    # Demo: Sinh và hiển thị dữ liệu mẫu
    print("=" * 60)
    print("Demo: Spot Price Generator")
    print("=" * 60)

    price_gen = SpotPriceGenerator()
    prices = price_gen.generate_price_series(24)  # 1 ngày
    print(f"Giá spot trong 24 giờ đầu tiên:")
    print(f"Min: ${min(prices):.4f}, Max: ${max(prices):.4f}, Mean: ${np.mean(prices):.4f}")

    print("\n" + "=" * 60)
    print("Demo: Workload Generator")
    print("=" * 60)

    for pattern in ["stable", "spike", "random", "periodic"]:
        wl_gen = WorkloadGenerator(pattern=pattern)
        workloads = wl_gen.generate_workload_series(24)
        print(f"\nPattern '{pattern}':")
        print(f"Min: {min(workloads)}, Max: {max(workloads)}, Mean: {np.mean(workloads):.1f}")

    print("\n" + "=" * 60)
    print("Demo: Sinh dữ liệu mẫu")
    print("=" * 60)

    df = generate_sample_data(n_days=1)
    print(df.head(10))
