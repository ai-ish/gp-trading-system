import numpy as np
import random
from deap import gp
import operator
from typing import Callable, List, Optional


class TradingPrimitives:
    """Trading primitives for genetic programming."""

    @staticmethod
    def setup_primitives():
        """Set up the primitive set for genetic programming."""
        # Define argument names
        args = ['close', 'open', 'high', 'low', 'volume']
        
        # Create primitive set
        pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray] * len(args), float)
        
        # Rename arguments
        for i, arg in enumerate(args):
            pset.renameArguments(**{f'ARG{i}': arg})
        
        # Store argument names in context
        pset.context = {'args': args}
        
        # Add arithmetic operators
        pset.addPrimitive(np.add, [float, float], float)
        pset.addPrimitive(np.subtract, [float, float], float)
        pset.addPrimitive(np.multiply, [float, float], float)
        pset.addPrimitive(TradingPrimitives.protected_division, [float, float], float)
        
        # Add logical operators
        pset.addPrimitive(TradingPrimitives.gt, [float, float], float)  # greater than
        pset.addPrimitive(TradingPrimitives.lt, [float, float], float)  # less than
        
        # Add conditional operator
        pset.addPrimitive(TradingPrimitives.if_then_else, [float, float, float], float)
        
        # Add technical indicators
        pset.addPrimitive(TradingPrimitives.sma, [np.ndarray, int], float)  # Simple Moving Average
        pset.addPrimitive(TradingPrimitives.ema, [np.ndarray, int], float)  # Exponential Moving Average
        pset.addPrimitive(TradingPrimitives.rsi, [np.ndarray, int], float)  # Relative Strength Index
        pset.addPrimitive(TradingPrimitives.bb_upper, [np.ndarray, int, float], float)  # Bollinger Band Upper
        pset.addPrimitive(TradingPrimitives.bb_lower, [np.ndarray, int, float], float)  # Bollinger Band Lower
        pset.addPrimitive(TradingPrimitives.macd, [np.ndarray, int, int], float)  # MACD
        
        # Add constants
        pset.addEphemeralConstant("randint", lambda: random.randint(2, 100), int)
        pset.addEphemeralConstant("randfloat", lambda: random.uniform(-1, 1), float)
        
        # Add specific constants that are commonly used
        for i in [5, 10, 20, 30, 50, 200]:
            pset.addTerminal(i, int)
        
        for i in [0.0, 1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.01, -0.01]:
            pset.addTerminal(i, float)
            
        return pset
    
    @staticmethod
    def protected_division(x, y):
        """Protected division to handle division by zero."""
        return x / y if abs(y) > 1e-6 else 0.0
    
    @staticmethod
    def gt(x, y):
        """Greater than operator that returns 1.0 for true, -1.0 for false."""
        return 1.0 if x > y else -1.0
    
    @staticmethod
    def lt(x, y):
        """Less than operator that returns 1.0 for true, -1.0 for false."""
        return 1.0 if x < y else -1.0
    
    @staticmethod
    def if_then_else(condition, output1, output2):
        """Conditional operator."""
        return output1 if condition > 0 else output2
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average."""
        # Ensure period is positive and not larger than the data
        period = max(1, min(period, len(data) - 1))
        return np.mean(data[-period:])
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average."""
        # Ensure period is positive and not larger than the data
        period = max(1, min(period, len(data) - 1))
        
        # Calculate EMA
        alpha = 2 / (period + 1)
        ema_value = data[-period]
        
        for i in range(-period + 1, 0):
            ema_value = alpha * data[i] + (1 - alpha) * ema_value
            
        return ema_value
    
    @staticmethod
    def rsi(data, period):
        """Relative Strength Index."""
        # Ensure period is positive and not larger than the data
        period = max(2, min(period, len(data) - 1))
        
        # Calculate price changes
        deltas = np.diff(data[-(period+1):])
        
        # Split gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        # Calculate averages
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Calculate RS and RSI
        if avg_loss < 1e-10:  # Avoid division by zero
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))
        
        return rsi_value
    
    @staticmethod
    def bb_upper(data, period, std_dev):
        """Bollinger Band Upper."""
        # Ensure period and std_dev are reasonable
        period = max(2, min(period, len(data) - 1))
        std_dev = max(0.1, min(std_dev, 10.0))
        
        # Calculate SMA and standard deviation
        sma = np.mean(data[-period:])
        sigma = np.std(data[-period:])
        
        # Calculate upper band
        upper_band = sma + std_dev * sigma
        
        return upper_band
    
    @staticmethod
    def bb_lower(data, period, std_dev):
        """Bollinger Band Lower."""
        # Ensure period and std_dev are reasonable
        period = max(2, min(period, len(data) - 1))
        std_dev = max(0.1, min(std_dev, 10.0))
        
        # Calculate SMA and standard deviation
        sma = np.mean(data[-period:])
        sigma = np.std(data[-period:])
        
        # Calculate lower band
        lower_band = sma - std_dev * sigma
        
        return lower_band
    
    @staticmethod
    def macd(data, fast_period, slow_period):
        """Moving Average Convergence Divergence."""
        # Ensure periods are reasonable
        fast_period = max(2, min(fast_period, len(data) - 2))
        slow_period = max(fast_period + 1, min(slow_period, len(data) - 1))
        
        # Calculate EMAs
        fast_ema = TradingPrimitives.ema(data, fast_period)
        slow_ema = TradingPrimitives.ema(data, slow_period)
        
        # Calculate MACD
        macd_value = fast_ema - slow_ema
        
        return macd_value