# <img src="docs/images/logo1.png" alt="icon" height="50" width="100" style="vertical-align:sub;"/> TFLlib: Trustworthy Federated Learning Library and Benchmark

---

TFLlib is a comprehensive library for trustworthy federated learning research based on [PFLlib](https://github.com/TsingZ0/PFLlib.git). It provides a unified framework to evaluate federated learning algorithms under various trustworthiness threats including backdoor attacks, Byzantine attacks, and privacy leakage.

![Framework](./docs/images/framework.png)

## Key Features

### Comprehensive FL Algorithms Support
- **Classic FL Algorithms**: FedAvg, FedProx, MOON, SCAFFOLD, FedDyn, FedNTD, FedGen
- **Extensible Architecture**: Easy to implement and integrate new FL algorithms

### Diverse Dataset Support
- **Computer Vision**: CIFAR-10, CIFAR-100, TinyImageNet, FEMNIST
- **Natural Language Processing**: IMDB, AGNews, Sent140
- **Tabular Data**: Adult, Heart, Credit Card, Texas100, Purchase100
- **Time Series**: UCI-HAR
- **Various Data Distribution Settings**: IID, Non-IID (Dirichlet, Pathological, etc.)

### Rich Model Zoo
- **CNN Models**: LeNet, SimpleCNN, ResNet series, VGG, MobileNet, ShuffleNet
- **NLP Models**: LSTM, BERT variants, ALBERT, ELECTRA, MobileBERT, MiniLM, TinyBERT
- **Other Models**: Logistic Regression, HAR-CNN, DeepSpeech

### Security Threats & Attacks

#### Poisoning Attacks
- **Backdoor Attacks**: [DBA](), [A3FL](), [CerP](), [EdgeCase](), [Neurotoxin](), [Replace]()
- **Byzantine Attacks**: [LIE](), [Fang](), [IPM](), [Label Flip](), [Median]() [Tailored](), [Min-Max](), [Noise](), [Sign Flip](), [SignGuard](), [Update Flip]()

#### Privacy Attacks
- **Membership Inference Attacks**: Nasr, Shokri, Zari, ML-Leaks
- **Label Inference Attacks**: Various LIA methods
- **Gradient Inversion Attacks**: DLG, Invert Gradients, See Through Gradients, LOKI, RobFed

### Defense Mechanisms
- **Robust Aggregation**: Krum, Bulyan, Coordinate-wise Median/Trimmed Mean, Geometric Median
- **Detection-Based**: Outlier Detection, Norm Difference Clipping, Cross-Round Defense
- **Others**: Soteria, Weak Differential Privacy, Robust Learning Rate, Foolsgold

### Real-world Simulation
- **System Heterogeneity**: Simulate varying computation capabilities of devices
- **Communication Heterogeneity**: Model unstable network conditions
- **Device Availability**: Handle dynamic client availability

### Multi-GPU Support
- Efficiently utilize multiple GPUs for large-scale federated learning simulations
- Accelerate both training and evaluation processes

## Architecture

```
TFLlib/
├── flcore/
│   ├── clients/              # Client-side implementations
│   ├── fedatasets/           # Federated datasets
│   │   ├── other/            # Various dataset implementations
│   │   └── utils/            # Dataset utilities
│   ├── models/               # Model architectures
│   ├── optimizers/           # Federated optimizers
│   ├── security/             # Security components
│   │   ├── attack/           # Various attack implementations
│   │   │   ├── poison/       # Poisoning attacks
│   │   │   └── privacy/      # Privacy attacks
│   │   └── defense/          # Defense mechanisms
│   ├── servers/              # Server-side implementations
│   ├── simulation/           # Real-world simulation modules
│   └── utils/                # Utility functions
├── main.py                   # Main entry point
├── run_exp_*.py              # Experiment scripts
└── config.py                 # Configuration parsing
```


## ToDo List

- ⭕️ Add the parameter configurations for each experiment script
- ⭕️ Provide .toml configuration files for easy experiment reproduction
- ⭕️ Polish the documentation and add more tutorials
- ⭕️ Provide datasets and pretrained models download scripts
- ⭕️ Add more defense mechanisms

## Getting Started [TODO]

### Installation

```bash
# Clone the repository
git clone https://github.com/xaddwell/TFLlib.git
cd TFLlib

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run federated learning experiments with various configurations:

```bash
# Basic FedAvg on CIFAR-10
python main.py --algorithm FedAvg --data_name CIFAR10 --model_name resnet18

# Run with non-IID data setting
python main.py --algorithm FedAvg --data_name CIFAR10 --model_name resnet18 --split_type diri --cncntrtn 0.5

# Run with system heterogeneity simulation
python main.py --algorithm FedAvg --data_name CIFAR10 --model_name resnet18 --dev_hetero 0.5 --comm_hetero 0.5
```

### Pre-configured Experiments

We provide several experiment scripts for reproducing results:

```bash
# Backdoor attack experiments
python run_exp_backdoor.py

# Byzantine attack experiments
python run_exp_byzantine.py

# Privacy attack experiments
python run_exp_inversion.py
python run_exp_lia.py
```

### Key Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `--algorithm` | FL algorithm to use | FedAvg |
| `--data_name` | Dataset to use | CIFAR10 |
| `--model_name` | Model architecture | resnet18 |
| `--num_clients` | Total number of clients | 100 |
| `--join_ratio` | Fraction of clients participating in each round | 0.1 |
| `--local_epochs` | Number of local training epochs | 2 |
| `--global_rounds` | Number of global communication rounds | 500 |
| `--split_type` | Data distribution type | iid |
| `--dev_hetero` | Device heterogeneity level (0-1) | 0.5 |
| `--comm_hetero` | Communication heterogeneity level (0-1) | 0.5 |

## Real-world Environment Simulation

TFLlib provides realistic simulation capabilities:

1. **Device Heterogeneity**: Clients have different computational capabilities
2. **Communication Heterogeneity**: Network conditions vary among clients
3. **Client Availability**: Dynamic client participation patterns

These features enable researchers to evaluate FL algorithms under practical deployment conditions.

## Multi-GPU Training

TFLlib supports multi-GPU training for accelerated experimentation:

```bash
python main.py --gpu 4  # Use 4 GPUs for training
```

## Citation

If you use TFLlib in your research, please cite our repo:

```bibtex
@misc{chen2025tfllib,
  title={TFLlib: Trustworthy Federated Learning Library and Benchmark},
  author={Jiahao Chen, Zhiming Zhao and Jianqing Zhang},
  year={2025},
  url={https://github.com/xaddwell/TFLlib}
}
```

## License

This project is licensed under the Apache License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments
We thank all the researchers who contribute to the development of TFLlib. Especially, we thank the benchmark [PFLlib](https://github.com/TsingZ0/PFLlib.git), provided by [Jianqing Zhang](https://github.com/TsingZ0).