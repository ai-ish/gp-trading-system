#!/usr/bin/env python
"""
Simple example of strategy evolution using the GP Trading System.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy.genetic.primitives import TradingPrimitives
from src.strategy.genetic.evolution import GeneticProgrammingEngine


def generate_sample_data(days=60):
    """Generate sample price data for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Generate random price series (with realistic properties)
    np.random.seed(42)
    
    # Start with a price and apply random walk
    initial_price = 50000.0  # Example BTC price
    returns = np.random.normal(0.0001, 0.01, size=len(dates))  # Small positive drift
    
    # Generate price series
    prices = initial_price * (1 + returns).cumprod()
    
    # Create OHLCV dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, size=len(dates))),
        'high': prices * (1 + np.random.normal(0.003, 0.003, size=len(dates))),
        'low': prices * (1 - np.random.normal(0.003, 0.003, size=len(dates))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, size=len(dates))
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Add some basic technical indicators
    # SMA
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df


def simple_fitness_function(individual):
    """
    Simple fitness function for evaluating strategies.
    
    This is just an example - a real fitness function would be more complex
    and consider more factors like risk-adjusted returns, drawdowns, etc.
    """
    from deap import gp
    
    # Compile the individual to a callable function
    pset = individual.pset
    func = gp.compile(individual, pset)
    
    # Get sample data
    df = generate_sample_data()
    
    # Initialize variables
    capital = 10000.0
    position = 0
    commission = 0.001  # 0.1% commission per trade
    
    # Convert dataframe to numpy arrays for faster processing
    close = df['close'].values
    open_prices = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Prepare arrays for other features the strategy might use
    feature_arrays = {}
    for col in df.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume']:
            feature_arrays[col] = df[col].values
    
    # Simulate trading
    equity_curve = [capital]
    trades = 0
    profitable_trades = 0
    
    for i in range(1, len(close)):
        # Prepare inputs for the strategy
        inputs = {
            'close': close[:i],
            'open': open_prices[:i],
            'high': high[:i],
            'low': low[:i],
            'volume': volume[:i],
            'position': position,
            'equity': equity_curve[-1]
        }
        
        # Add other features
        for name, values in feature_arrays.items():
            inputs[name] = values[:i]
        
        # Get signal from strategy (-1 for short, 0 for flat, 1 for long)
        try:
            signal_value = func(*[inputs[arg] for arg in pset.context['args']])
            
            if signal_value > 0.2:
                signal = 1
            elif signal_value < -0.2:
                signal = -1
            else:
                signal = 0
                
        except Exception:
            # If strategy fails, provide a bad fitness
            return (-100.0,)
        
        # Process trading logic
        if position != signal:
            # Close existing position
            if position != 0:
                # Calculate P&L
                if position == 1:
                    profit = (close[i] - entry_price) / entry_price
                else:  # position == -1
                    profit = (entry_price - close[i]) / entry_price
                
                # Apply commission
                profit -= commission
                
                # Update capital
                capital *= (1 + profit * leverage)
                
                # Track trades
                trades += 1
                if profit > 0:
                    profitable_trades += 1
            
            # Open new position
            if signal != 0:
                entry_price = close[i]
                position = signal
            else:
                position = 0
        
        # Update equity curve
        equity_curve.append(capital)
    
    # Calculate fitness metrics
    final_return = (capital - 10000) / 10000
    
    # Penalize strategies with too few trades
    if trades < 5:
        return (final_return * 0.5,)
    
    # Calculate Sharpe ratio (simple version)
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    # Win rate
    win_rate = profitable_trades / trades if trades > 0 else 0
    
    # Combine metrics for final fitness (example weighting)
    fitness = 0.4 * final_return + 0.4 * sharpe_ratio + 0.2 * win_rate
    
    return (fitness,)


def run_example():
    """Run a simple strategy evolution example."""
    print("Starting Simple GP Trading Strategy Evolution Example")
    
    # Setup primitive set
    pset = TradingPrimitives.setup_primitives()
    
    # Setup GP engine
    gp_engine = GeneticProgrammingEngine(
        pset=pset,
        population_size=100,  # Small population for quick example
        tournament_size=5,
        crossover_prob=0.7,
        mutation_prob=0.3,
        max_depth=5  # Limit depth for simpler strategies
    )
    
    # Run evolution
    best_strategy, stats = gp_engine.evolve(
        fitness_func=simple_fitness_function,
        generations=10,  # Few generations for quick example
        hall_of_fame_size=5
    )
    
    # Print best strategy
    print("\nBest Strategy:")
    print(str(best_strategy))
    
    # Compile and test the best strategy
    strategy_func = gp_engine.compile_strategy(best_strategy, pset)
    
    # Generate test data
    test_data = generate_sample_data(days=30)  # New data for out-of-sample testing
    
    # Backtest the strategy
    print("\nBacktesting best strategy on new data...")
    
    # Convert dataframe to numpy for faster processing
    close = test_data['close'].values
    
    # Prepare other arrays the strategy might need
    inputs = {}
    for col in test_data.columns:
        inputs[col] = test_data[col].values
    
    # Initialize trading variables
    capital = 10000.0
    position = 0
    entry_price = 0
    commission = 0.001
    leverage = 1
    
    # Arrays to store results
    equity_curve = [capital]
    positions = [0]
    
    # Run backtest
    for i in range(1, len(close)):
        # Prepare inputs for the strategy decision
        strategy_inputs = {}
        for col in inputs:
            strategy_inputs[col] = inputs[col][:i]
        
        # Get signal from strategy
        try:
            signal_value = strategy_func(*[strategy_inputs[arg] for arg in pset.context['args']])
            
            if signal_value > 0.2:
                signal = 1
            elif signal_value < -0.2:
                signal = -1
            else:
                signal = 0
                
        except Exception as e:
            print(f"Strategy error: {e}")
            signal = 0
        
        # Process trading logic
        if position != signal:
            # Close existing position
            if position != 0:
                # Calculate P&L
                if position == 1:
                    profit = (close[i] - entry_price) / entry_price
                else:  # position == -1
                    profit = (entry_price - close[i]) / entry_price
                
                # Apply commission
                profit -= commission
                
                # Update capital
                capital *= (1 + profit * leverage)
            
            # Open new position
            if signal != 0:
                entry_price = close[i]
                position = signal
            else:
                position = 0
        
        # Update tracking variables
        equity_curve.append(capital)
        positions.append(position)
    
    # Calculate performance metrics
    final_return = (capital - 10000) / 10000
    print(f"Final Return: {final_return:.2%}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price chart
    ax1.plot(test_data.index, test_data['close'], label='BTC Price')
    ax1.set_title('Price Chart')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot equity curve
    ax2.plot(test_data.index, equity_curve, label='Equity Curve')
    ax2.set_title('Strategy Performance')
    ax2.set_ylabel('Equity')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('examples/strategy_results.png')
    plt.close()
    
    print(f"Results saved to examples/strategy_results.png")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    # Run the example
    run_example()