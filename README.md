# Genetic Programming Trading System

This repository contains a professional Bitcoin Futures Algorithmic Trading System using Genetic Programming. The system is designed for developing and backtesting trading strategies using evolutionary algorithms.

## Project Overview

The system implements:
- Genetic programming for evolving trading strategies
- Backtesting framework for strategy evaluation
- Data acquisition and preprocessing for Bitcoin futures
- Risk management components
- Performance analysis tools

## Project Structure

```
gp-trading-system/
├── src/                       # Core implementation
│   ├── data/                  # Data acquisition and processing
│   ├── strategy/              # Strategy development with GP
│   ├── backtest/              # Backtesting engine
│   ├── risk/                  # Risk management
│   ├── execution/             # Trade execution
│   └── monitoring/            # System monitoring
├── tests/                     # Unit and integration tests
├── notebooks/                 # Research notebooks
├── configs/                   # Configuration files
├── requirements.txt           # Python dependencies
└── main.py                    # Main entry point
```

## Requirements

The system requires several Python libraries:

```
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
seaborn==0.12.2
deap==1.4.1
h5py==3.8.0
tables==3.8.0
scikit-learn==1.2.2
plotly==5.15.0
```

See `requirements.txt` for the complete list of dependencies.

## Usage

To run the main genetic programming system:

```bash
python main.py
```

For backtesting a specific strategy:

```bash
python backtest_strategy.py
```

## Testing

The system includes various test scripts for validating functionality:

```bash
./run_correctness_tests.sh
```