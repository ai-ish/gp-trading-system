import argparse
import sys
import os
import pickle
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from src.data.fetchers import CCXTDataFetcher
from src.data.preprocessors import FeatureEngineer
from src.data.database import TimeSeriesDatabase
from src.strategy.genetic.primitives import TradingPrimitives
from src.strategy.genetic.evolution import GeneticProgrammingEngine
from src.strategy.genetic.fitness import StrategyFitnessEvaluator
from src.backtest.engine import BacktestEngine
from src.backtest.visualizers import create_performance_report
from src.risk.position_sizing import RiskManager
from loguru import logger

class AlgoTradingSystem:
    def __init__(self,
                 config_path: str = 'configs/config.json',
                 mode: str = 'backtest'):
        """Initialize the algorithmic trading system."""
        self.config = self._load_config(config_path)
        self.mode = mode  # 'backtest', 'paper', or 'live'

        # Initialize components based on config
        self.data_fetcher = CCXTDataFetcher(
            exchange_id=self.config['exchange']['id'],
            api_key=self.config['exchange'].get('api_key'),
            secret=self.config['exchange'].get('secret')
        )

        self.feature_engineer = FeatureEngineer()
        self.db = TimeSeriesDatabase(self.config['data']['db_path'])

        # Initialize trading strategy components
        self.pset = TradingPrimitives.setup_primitives()

        # Initialize risk manager
        self.risk_manager = RiskManager(
            capital=self.config['capital'],
            max_position_size=self.config['risk']['max_position_size'],
            max_drawdown=self.config['risk']['max_drawdown'],
            volatility_target=self.config['risk']['volatility_target']
        )

        # Setup logger
        self._setup_logger()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _setup_logger(self):
        """Configure the logger."""
        log_path = self.config['logging']['path']
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logger.remove()  # Remove default logger
        logger.add(sys.stderr, level=self.config['logging']['console_level'])
        logger.add(log_path,
                  rotation=self.config['logging']['rotation'],
                  level=self.config['logging']['file_level'])

    def fetch_and_prepare_data(self, symbol: str, timeframe: str, days_back: int = 30) -> dict:
        """Fetch and prepare data for trading or backtesting."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        # Try to load from database first
        df = self.db.load_dataframe(symbol, timeframe)

        if df is not None and len(df) > 0 and df.index[-1].date() >= (end_date - timedelta(days=1)).date():
            logger.info(f"Using cached data from database")
        else:
            # Fetch new data
            try:
                df = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                # Store in database
                if df is not None and len(df) > 0:
                    self.db.store_dataframe(df, symbol, timeframe)
            except Exception as e:
                logger.error(f"Error fetching data from {self.config['exchange']['id']}: {str(e)}")
                
                # Try Alpha Vantage as a fallback for real data
                try:
                    logger.info(f"Attempting to fetch data from Alpha Vantage...")
                    df = self._fetch_alpha_vantage_data(symbol, timeframe, start_date, end_date)
                    
                    # Store in database if successful
                    if df is not None and len(df) > 0:
                        self.db.store_dataframe(df, symbol, timeframe)
                        logger.info(f"Successfully fetched data from Alpha Vantage")
                except Exception as av_error:
                    logger.error(f"Error fetching data from Alpha Vantage: {str(av_error)}")
                    logger.info(f"Falling back to sample data...")
                    df = self._generate_sample_data(symbol, timeframe, start_date, end_date)
            
        # Add features
        if df is not None and len(df) > 0:
            df = self.feature_engineer.add_technical_indicators(df)
        else:
            logger.error(f"No data available for {symbol} with timeframe {timeframe}")
            raise ValueError(f"No data available for {symbol} with timeframe {timeframe}")

        return df

    def train_strategy(self, market_data, population_size: int = 300, generations: int = 50) -> dict:
        """Train a trading strategy using genetic programming."""
        logger.info(f"Starting GP training with pop={population_size}, gen={generations}")
        
        # Configure the fitness evaluator
        fitness_evaluator = StrategyFitnessEvaluator(
            market_data=market_data,
            capital=self.config['capital'],
            commission=self.config['exchange']['commission'],
            leverage=self.config['trading']['leverage']
        )
        
        # Configure the GP engine
        gp_engine = GeneticProgrammingEngine(
            pset=self.pset,
            population_size=population_size,
            tournament_size=self.config['gp']['tournament_size'],
            crossover_prob=self.config['gp']['crossover_prob'],
            mutation_prob=self.config['gp']['mutation_prob'],
            max_depth=self.config['gp']['max_depth']
        )
        
        # Inject seed strategies
        self._create_seed_strategies(gp_engine)
        
        # Run evolution
        start_time = time.time()
        best_strategy, stats = gp_engine.evolve(
            fitness_func=fitness_evaluator.evaluate,
            generations=generations,
            hall_of_fame_size=self.config['gp']['hall_of_fame_size']
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save the best strategy
        strategy_path = self._save_strategy(best_strategy)
        
        # Backtest the best strategy
        backtest_results = self.backtest_strategy(best_strategy, market_data)
        
        # Save backtest results
        results_path = self._save_backtest_results(backtest_results)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Strategy saved to {strategy_path}")
        logger.info(f"Backtest results saved to {results_path}")
        
        return {
            "best_strategy": best_strategy,
            "stats": stats,
            "backtest_results": backtest_results,
            "training_time": training_time
        }

    def backtest_strategy(self, strategy, market_data) -> dict:
        """Backtest a trading strategy."""
        logger.info("Starting backtest")
        
        # Configure the backtest engine
        backtest_engine = BacktestEngine(
            capital=self.config['capital'],
            commission=self.config['exchange']['commission'],
            leverage=self.config['trading']['leverage'],
            risk_manager=self.risk_manager
        )
        
        # Run the backtest
        results = backtest_engine.run(
            strategy=self._strategy_adapter(strategy, {}),
            market_data=market_data
        )
        
        logger.info(f"Backtest completed with final equity: {results['final_equity']:.2f}")
        
        return results

    def _save_strategy(self, individual, suffix=""):
        """Save a strategy to disk."""
        results_dir = self.config['results']['directory']
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_{timestamp}{suffix}.pkl"
        path = os.path.join(results_dir, filename)
        
        with open(path, 'wb') as f:
            pickle.dump(individual, f)
            
        return path

    def _strategy_adapter(self, strategy_func, context):
        """Adapt the GP strategy to the format expected by backtest engine."""
        def adapted_strategy(data, position=None, equity=None):
            # Create a dict with all inputs the strategy might need
            inputs = {
                'open': data['open'].values,
                'high': data['high'].values,
                'low': data['low'].values,
                'close': data['close'].values,
                'volume': data['volume'].values,
                'position': 0 if position is None else position,
                'equity': 10000 if equity is None else equity
            }
            
            # Add all technical indicators
            for col in data.columns:
                if col not in inputs:
                    inputs[col] = data[col].values
                    
            # Call the strategy function with the inputs
            signal = strategy_func(*[inputs[arg] for arg in strategy_func.pset.context['args']])
            
            # Return -1, 0, or 1 for short, flat, or long
            if signal < -0.2:
                return -1
            elif signal > 0.2:
                return 1
            else:
                return 0
                
        return adapted_strategy

    def _create_seed_strategies(self, gp_engine):
        """Create seed strategies to inject into the initial population."""
        # Simple Moving Average Crossover
        def create_sma_crossover():
            from deap import gp
            import operator
            
            # SMA 10 crosses above SMA 30
            return gp.PrimitiveTree([
                gp_engine.pset.mapping["gt"],
                gp_engine.pset.mapping["sma"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 10
                gp_engine.pset.mapping["sma"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 30
            ])
            
        # RSI Strategy
        def create_rsi_strategy():
            from deap import gp
            
            # Buy when RSI < 30, sell when RSI > 70
            return gp.PrimitiveTree([
                gp_engine.pset.mapping["if_then_else"],
                gp_engine.pset.mapping["lt"],
                gp_engine.pset.mapping["rsi"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 14
                gp_engine.pset.mapping["ephemeral"](),  # 30
                gp_engine.pset.mapping["ephemeral"](),  # 1 (buy)
                gp_engine.pset.mapping["if_then_else"],
                gp_engine.pset.mapping["gt"],
                gp_engine.pset.mapping["rsi"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 14
                gp_engine.pset.mapping["ephemeral"](),  # 70
                gp_engine.pset.mapping["ephemeral"](),  # -1 (sell)
                gp_engine.pset.mapping["ephemeral"](),  # 0 (hold)
            ])
            
        # Bollinger Band Strategy
        def create_bb_strategy():
            from deap import gp
            
            # Buy when price touches lower band, sell when it touches upper band
            return gp.PrimitiveTree([
                gp_engine.pset.mapping["if_then_else"],
                gp_engine.pset.mapping["lt"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["bb_lower"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 20
                gp_engine.pset.mapping["ephemeral"](),  # 2
                gp_engine.pset.mapping["ephemeral"](),  # 1 (buy)
                gp_engine.pset.mapping["if_then_else"],
                gp_engine.pset.mapping["gt"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["bb_upper"],
                gp_engine.pset.terminals[0],  # close
                gp_engine.pset.mapping["ephemeral"](),  # 20
                gp_engine.pset.mapping["ephemeral"](),  # 2
                gp_engine.pset.mapping["ephemeral"](),  # -1 (sell)
                gp_engine.pset.mapping["ephemeral"](),  # 0 (hold)
            ])
        
        # Create seed strategies and inject them
        try:
            seed_strategies = [
                create_sma_crossover(),
                create_rsi_strategy(),
                create_bb_strategy()
            ]
            
            # Inject seed strategies into initial population
            gp_engine.seed_strategies = seed_strategies
            logger.info(f"Added {len(seed_strategies)} seed strategies to the initial population")
        except Exception as e:
            logger.error(f"Error creating seed strategies: {e}")

    def _save_backtest_results(self, results):
        """Save backtest results to disk."""
        results_dir = self.config['results']['directory']
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        path = os.path.join(results_dir, filename)
        
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        return path

    def load_strategy(self, strategy_path: str):
        """Load a saved strategy from disk."""
        with open(strategy_path, 'rb') as f:
            strategy = pickle.load(f)
        
        logger.info(f"Loaded strategy from {strategy_path}")
        return strategy


def visualize_backtest_results(results, symbol):
    """Create visualizations of backtest results."""
    # Extract data from results
    timestamps = results['timestamps']
    equity_curve = results['equity_curve']
    positions = results['positions']
    
    # Convert to datetime if needed
    if isinstance(timestamps[0], str):
        timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
    
    # Create equity curve plot
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, equity_curve, label='Equity Curve')
    
    # Add position entry/exit markers
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            marker = '^' if positions[i] > positions[i-1] else 'v'
            color = 'g' if positions[i] > positions[i-1] else 'r'
            plt.scatter(timestamps[i], equity_curve[i], marker=marker, color=color, s=100)
    
    plt.title(f'Backtest Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(f'results/backtest_{symbol}_{datetime.now().strftime("%Y%m%d")}.png')
    plt.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Algorithmic Trading System')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default='backtest',
                        help='Trading mode: backtest, paper, or live')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Trading timeframe')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data')
    parser.add_argument('--config', default='configs/config.json', help='Path to config file')
    parser.add_argument('--population', type=int, default=300, help='GP population size')
    parser.add_argument('--generations', type=int, default=50, help='GP generations')
    parser.add_argument('--strategy', help='Path to saved strategy (skips training)')
    
    args = parser.parse_args()
    
    # Initialize the trading system
    system = AlgoTradingSystem(config_path=args.config, mode=args.mode)
    
    # Fetch and prepare data
    market_data = system.fetch_and_prepare_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days_back=args.days
    )
    
    # Use saved strategy or train a new one
    if args.strategy:
        strategy = system.load_strategy(args.strategy)
    else:
        results = system.train_strategy(
            market_data=market_data,
            population_size=args.population,
            generations=args.generations
        )
        strategy = results['best_strategy']
    
    # Backtest the strategy
    backtest_results = system.backtest_strategy(strategy, market_data)
    
    # Visualize the results
    visualize_backtest_results(backtest_results, args.symbol)
    
    # Create performance report
    create_performance_report(backtest_results, args.symbol)


if __name__ == "__main__":
    main()