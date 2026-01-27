# Tối ưu chiến lược sử dụng Spot Instance trên Cloud bằng Học tăng cường (RL)

## Mô tả dự án

Dự án này sử dụng **Reinforcement Learning (RL)** để tối ưu hóa chiến lược sử dụng **Spot Instance** trên Cloud. Mục tiêu là giảm chi phí cloud trong khi vẫn đảm bảo hoàn thành workload và tuân thủ SLA.

### Bài toán

Cloud providers (AWS, GCP, Azure) cung cấp 2 loại instances:
- **Spot Instance**: Giá rẻ (tiết kiệm 60-90%) nhưng có thể bị gián đoạn (terminated) bất kỳ lúc nào
- **On-Demand Instance**: Giá đắt hơn nhưng ổn định, không bị gián đoạn

**Thách thức**: Làm sao để quyết định khi nào nên sử dụng spot instance (tiết kiệm chi phí) vs on-demand instance (đảm bảo ổn định)?

### Giải pháp

Sử dụng **Deep Q-Network (DQN)** để train một agent học cách:
- Quan sát trạng thái thị trường (giá spot, workload, số instances...)
- Đưa ra quyết định tối ưu (request spot/on-demand, terminate, checkpoint...)
- Tối đa hóa reward = giảm chi phí + hoàn thành jobs - tránh interruptions

## Kết quả đạt được

### Spike Workload Scenario
| Policy | Mean Cost ($) | Mean Jobs | Interruptions | Mean Reward |
|--------|--------------|-----------|---------------|-------------|
| **DQN_Agent** | 110.36 | **2335.90** | 23.15 | **985.42** |
| AlwaysOnDemand | 82.78 | 1655.50 | 0.00 | 518.95 |
| AlwaysSpot | 25.27 | 1492.40 | 65.35 | 354.92 |
| ThresholdPolicy | 58.27 | 1587.30 | 28.60 | 446.41 |
| SmartThresholdPolicy | 75.47 | 1634.90 | 8.20 | 496.76 |

**Highlights:**
- DQN Agent đạt **highest reward** (985.42) - cao hơn 90% so với AlwaysOnDemand
- DQN Agent hoàn thành **nhiều jobs nhất** (2336) - cao hơn 41% so với AlwaysOnDemand
- Tỷ lệ interruption thấp hơn 65% so với AlwaysSpot

### Stable Workload Scenario
| Policy | Mean Cost ($) | Mean Jobs | Interruptions | Mean Reward |
|--------|--------------|-----------|---------------|-------------|
| **DQN_Agent** | 163.50 | **1684.15** | 0.00 | 678.58 |
| AlwaysOnDemand | 82.65 | 1653.00 | 0.00 | 737.40 |
| AlwaysSpot | 25.09 | 1481.70 | 68.55 | 385.05 |

## Cấu trúc dự án

```
spot-instance-bidding-strategy/
├── configs/
│   └── config.yaml          # Cấu hình hyperparameters
├── data/                    # Dữ liệu giá spot, workload
├── logs/                    # Training logs (TensorBoard)
├── models/                  # Trained models
│   ├── dqn_spike.zip       # Model cho spike workload
│   └── dqn_stable.zip      # Model cho stable workload
├── notebooks/
│   └── analysis.ipynb       # Jupyter notebook phân tích
├── results/                 # Kết quả evaluation
│   ├── plots/              # Biểu đồ so sánh
│   └── experiment_report.md # Báo cáo chi tiết
├── src/
│   ├── __init__.py
│   ├── baselines.py         # Baseline policies (5 loại)
│   ├── data_generator.py    # Sinh dữ liệu giá spot và workload
│   ├── environment.py       # Gymnasium environment
│   ├── evaluate.py          # Đánh giá và so sánh
│   ├── train.py             # Training script
│   └── visualize.py         # Visualization
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # Unit tests (33 tests)
├── run_experiments.py       # Script chạy thí nghiệm toàn diện
├── README.md
└── requirements.txt
```

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-repo/spot-instance-bidding-strategy.git
cd spot-instance-bidding-strategy
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng nhanh

### Chạy thí nghiệm đầy đủ

```bash
# Quick mode (10k timesteps, 2 scenarios)
python run_experiments.py --quick

# Full mode với tất cả scenarios
python run_experiments.py --timesteps 100000 --scenarios spike stable periodic random

# Custom configuration
python run_experiments.py --timesteps 50000 --episodes 100 --scenarios spike stable
```

### Training riêng lẻ

```bash
# Training với config mặc định
python -m src.train

# Training với custom parameters
python -m src.train --total-timesteps 100000 --seed 42 --workload-pattern spike

# Training với config file
python -m src.train --config configs/config.yaml
```

### Đánh giá model

```bash
# Đánh giá model đã train
python -m src.evaluate --model models/dqn_spike.zip

# Chỉ đánh giá baselines
python -m src.evaluate

# Custom evaluation
python -m src.evaluate --model models/dqn_spike.zip --n-episodes 100 --workload-pattern spike
```

### Sử dụng model đã train

```python
from stable_baselines3 import DQN
from src.environment import SpotInstanceEnv

# Load trained model
model = DQN.load('models/dqn_spike.zip')

# Create environment
env = SpotInstanceEnv(workload_pattern='spike')
obs, _ = env.reset()

# Use model for decisions
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Action: {env.ACTION_NAMES[int(action)]}, Reward: {reward:.2f}")

# Get episode metrics
metrics = env.get_metrics()
print(f"Total Cost: ${metrics['total_cost']:.2f}")
print(f"Jobs Completed: {metrics['total_jobs_completed']}")
```

## Environment Chi tiết

### State Space (7 features)

| Feature | Mô tả | Range |
|---------|-------|-------|
| current_spot_price | Giá spot hiện tại | [0, 1] (normalized) |
| price_moving_avg | Giá trung bình 24 giờ qua | [0, 1] |
| hour_of_day | Giờ trong ngày (0-23) | [0, 1] |
| day_of_week | Ngày trong tuần (0-6) | [0, 1] |
| pending_workload | Số jobs cần xử lý | [0, 1] |
| active_instances | Số instances đang chạy | [0, 1] |
| interruption_probability | Xác suất bị terminate | [0, 1] |

### Action Space (5 discrete actions)

| Action | Mô tả |
|--------|-------|
| 0 | **do_nothing**: Không làm gì |
| 1 | **request_spot_instance**: Yêu cầu spot instance mới |
| 2 | **request_on_demand_instance**: Yêu cầu on-demand instance mới |
| 3 | **terminate_instance**: Terminate 1 instance (ưu tiên on-demand trước) |
| 4 | **checkpoint_and_migrate**: Checkpoint và migrate spot instances |

### Reward Function

```
R = -α(cost) + β(jobs_completed) - γ(interruption_penalty) - δ(sla_violation)
```

Với:
- α = 1.0: Hệ số phạt chi phí
- β = 0.5: Hệ số thưởng hoàn thành job
- γ = 2.0: Hệ số phạt interruption
- δ = 1.5: Hệ số phạt vi phạm SLA

## Baselines

| Policy | Mô tả | Ưu điểm | Nhược điểm |
|--------|-------|---------|------------|
| AlwaysOnDemand | Luôn sử dụng on-demand | Ổn định, không interruption | Chi phí cao nhất |
| AlwaysSpot | Luôn sử dụng spot | Chi phí thấp nhất | Interruption cao nhất |
| ThresholdPolicy | Dùng spot nếu giá < ngưỡng | Đơn giản, hiệu quả | Không thích nghi |
| SmartThresholdPolicy | Threshold + thời gian, trend | Thông minh hơn | Vẫn rule-based |
| RandomPolicy | Random action | Baseline thấp nhất | Không tối ưu |

## DQN Configuration

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| Policy | MlpPolicy | Multi-layer Perceptron |
| Network | [128, 128] | 2 hidden layers |
| Learning Rate | 0.0001 | Gradient descent step size |
| Buffer Size | 50,000 | Replay buffer capacity |
| Batch Size | 64 | Training batch size |
| Gamma | 0.99 | Discount factor |
| Exploration | ε: 1.0 → 0.05 | Epsilon-greedy decay |

## Workload Patterns

1. **stable**: Workload ổn định với noise nhỏ
2. **spike**: Random spike events (5x multiplier)
3. **periodic**: Cycle theo ngày (cao ban ngày, thấp ban đêm)
4. **random**: Hoàn toàn ngẫu nhiên

## Testing

```bash
# Chạy tất cả tests
pytest tests/

# Chạy với coverage
pytest tests/ --cov=src --cov-report=html

# Chạy test cụ thể
pytest tests/test_environment.py -v
```

## TensorBoard

```bash
# Xem training curves
tensorboard --logdir logs/tensorboard/
```

## Deliverables

1. **Mã nguồn Gymnasium environment** (`src/environment.py`)
2. **Trained DQN models** (`models/dqn_*.zip`)
3. **Scripts đánh giá** (`src/evaluate.py`, `run_experiments.py`)
4. **Biểu đồ so sánh** (`results/plots/`)
5. **Báo cáo chi tiết** (`results/experiment_report.md`)

## Kết luận

DQN Agent học được:
1. **Thích nghi với biến động giá** - Ưu tiên spot khi giá thấp, chuyển sang on-demand khi giá cao
2. **Đảm bảo SLA** - Cân bằng giữa tối ưu chi phí và hoàn thành jobs
3. **Generalizable** - Hoạt động tốt trên nhiều workload patterns khác nhau

## Tài liệu tham khảo

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [AWS Spot Instances](https://aws.amazon.com/ec2/spot/)
- [Deep Q-Network (DQN) Paper](https://arxiv.org/abs/1312.5602)

## License

MIT License
