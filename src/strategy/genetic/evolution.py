import random
import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional

from deap import base, creator, tools, gp, algorithms
from loguru import logger


class GeneticProgrammingEngine:
    """Engine for evolving trading strategies using genetic programming."""
    
    def __init__(self, 
                 pset: gp.PrimitiveSetTyped,
                 population_size: int = 300,
                 tournament_size: int = 7,
                 crossover_prob: float = 0.9, 
                 mutation_prob: float = 0.1,
                 max_depth: int = 8):
        """Initialize the GP engine with parameters."""
        self.pset = pset
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.seed_strategies = []
        
        # Set up DEAP creator if not done already
        self._setup_creator()
        
        # Set up toolbox
        self.toolbox = self._setup_toolbox()
        
    def _setup_creator(self):
        """Set up the DEAP creator with fitness and individual classes."""
        # Check if already registered to avoid errors
        if 'FitnessMax' not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        if 'Individual' not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    def _setup_toolbox(self) -> base.Toolbox:
        """Set up the DEAP toolbox with genetic operators."""
        toolbox = base.Toolbox()
        
        # Register expression generation
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=self.max_depth)
        
        # Register individual and population creation
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=self.max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        
        # Set depth limit for crossover and mutation
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        
        return toolbox
    
    def evolve(self, 
               fitness_func: Callable,
               generations: int = 50,
               hall_of_fame_size: int = 10) -> Tuple[Any, Dict]:
        """Run the genetic programming evolution process."""
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # If we have seed strategies, inject them
        if self.seed_strategies:
            for i, strategy in enumerate(self.seed_strategies):
                if i < len(pop):
                    pop[i] = creator.Individual(strategy)
                    
        # Create Hall of Fame to keep track of best individuals
        hof = tools.HallOfFame(hall_of_fame_size)
        
        # Create statistics to track evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Register the fitness evaluation function
        self.toolbox.register("evaluate", fitness_func)
        
        # Run the evolution
        logger.info(f"Starting evolution with population size {self.population_size} for {generations} generations")
        
        # Run evolution with logging
        for gen in range(generations):
            # Check for early termination
            if gen > 0 and hof[0].fitness.values[0] >= 0.99:
                logger.info(f"Early termination at generation {gen} with fitness {hof[0].fitness.values[0]}")
                break
                
            # Generate offspring using tournament selection, crossover, and mutation
            offspring = algorithms.varAnd(
                pop, 
                self.toolbox, 
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob
            )
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update population
            pop[:] = offspring
            
            # Update Hall of Fame
            hof.update(pop)
            
            # Compile statistics
            record = stats.compile(pop)
            
            # Log progress
            logger.info(f"Gen {gen}: Max = {record['max']:.4f}, Avg = {record['avg']:.4f}, Min = {record['min']:.4f}, Std = {record['std']:.4f}")
            
        # Return best individual and stats
        logger.info(f"Evolution completed with best fitness: {hof[0].fitness.values[0]}")
        return hof[0], {"hall_of_fame": hof, "stats": stats}
    
    @staticmethod
    def get_human_readable_strategy(individual, pset):
        """Get a human-readable expression of the strategy."""
        return str(gp.compile(individual, pset))
    
    @staticmethod
    def compile_strategy(individual, pset):
        """Compile the strategy to a callable function."""
        return gp.compile(individual, pset)


# Import here to avoid circular imports
import operator