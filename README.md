# Tối ưu chiến lược sử dụng Spot Instance trên Cloud bằng Học tăng cường (RL)

## Mô tả dự án

Dự án này sử dụng **Reinforcement Learning (RL)** để tối ưu hóa chiến lược sử dụng **Spot Instance** trên Cloud. Mục tiêu là giảm chi phí cloud trong khi vẫn đảm bảo hoàn thành workload.

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

## Cấu trúc dự án

```
spot-instance-bidding-strategy/
├── configs/
│   └── config.yaml          # Cấu hình hyperparameters
├── data/                    # Dữ liệu giá spot, workload
├── docs/                    # Tài liệu chi tiết
├── logs/                    # Training logs
├── models/                  # Trained models
├── notebooks/
│   └── analysis.ipynb       # Jupyter notebook phân tích
├── results/                 # Kết quả evaluation
├── src/
│   ├── __init__.py
│   ├── baselines.py         # Baseline policies
│   ├── data_generator.py    # Sinh dữ liệu giá spot và workload
│   ├── environment.py       # Gymnasium environment
│   ├── evaluate.py          # Đánh giá và so sánh
│   ├── train.py             # Training script
│   └── visualize.py         # Visualization
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # Unit tests
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

## Sử dụng

### 1. Training RL Agent

```bash
# Training với config mặc định
python -m src.train

# Training với custom parameters
python -m src.train --total-timesteps 100000 --seed 42 --workload-pattern spike

# Training với config file
python -m src.train --config configs/config.yaml
```

**Các options:**
- `--config, -c`: Đường dẫn file config YAML
- `--total-timesteps, -t`: Tổng số timesteps training (default: 100000)
- `--seed, -s`: Random seed (default: 42)
- `--workload-pattern, -w`: Pattern workload (stable/spike/random/periodic)
- `--save-path`: Đường dẫn lưu model
- `--log-path`: Đường dẫn log TensorBoard
- `--verbose, -v`: Mức độ verbose (0/1/2)

### 2. Đánh giá và so sánh

```bash
# Đánh giá model đã train
python -m src.evaluate --model models/best_model.zip

# Chỉ đánh giá baselines
python -m src.evaluate

# Custom evaluation
python -m src.evaluate --model models/best_model.zip --n-episodes 100 --workload-pattern spike
```

**Các options:**
- `--model, -m`: Đường dẫn trained model
- `--n-episodes, -n`: Số episodes để evaluate (default: 100)
- `--workload-pattern, -w`: Pattern workload
- `--save-path`: Đường dẫn lưu kết quả
- `--no-plot`: Không vẽ plots

### 3. Visualization

```bash
# Chạy visualization module
python -m src.visualize
```

### 4. TensorBoard

```bash
# Xem training curves
tensorboard --logdir logs/tensorboard/
```

## Environment

### State Space (7 features)

| Feature | Mô tả | Range |
|---------|-------|-------|
| current_spot_price | Giá spot hiện tại | [0, 1] (normalized) |
| price_moving_avg | Giá trung bình 1 giờ qua | [0, 1] |
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
| 3 | **terminate_instance**: Terminate 1 instance |
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

| Policy | Mô tả |
|--------|-------|
| AlwaysOnDemand | Luôn sử dụng on-demand instances |
| AlwaysSpot | Luôn sử dụng spot instances |
| ThresholdPolicy | Dùng spot nếu giá < ngưỡng |
| SmartThresholdPolicy | Threshold với xem xét thời gian, trend |
| RandomPolicy | Random action |

## Kết quả mong đợi

Sau khi training, RL agent sẽ:
- Giảm **20-40% chi phí** so với AlwaysOnDemand
- Giảm **50-70% interruptions** so với AlwaysSpot
- Tối ưu **cost per job** tốt hơn tất cả baselines

## Cấu hình

File `configs/config.yaml` chứa các hyperparameters:

```yaml
# Environment
environment:
  on_demand_price: 0.10
  spot_price_base: 0.03
  max_instances: 10
  episode_length: 168

# DQN
dqn:
  learning_rate: 0.0001
  buffer_size: 100000
  batch_size: 64
  gamma: 0.99

# Training
training:
  total_timesteps: 100000
  eval_freq: 5000
```

## Testing

```bash
# Chạy tất cả tests
pytest tests/

# Chạy với coverage
pytest tests/ --cov=src --cov-report=html
```

## Tài liệu tham khảo

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [AWS Spot Instances](https://aws.amazon.com/ec2/spot/)
- [Deep Q-Network (DQN) Paper](https://arxiv.org/abs/1312.5602)

## License

MIT License
