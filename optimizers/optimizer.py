"""
MPC 优化算法实现
支持: 粒子群优化 (PSO)、模拟退火 (SA)、差分进化 (DE)
"""
import numpy as np
from typing import Callable, Tuple
from config import MPCConfig


class BaseOptimizer:
    """优化器基类"""
    
    def __init__(self, config: MPCConfig):
        self.config = config
        self.control_horizon = config.control_horizon
        self.dose_min = config.dose_min
        self.dose_max = config.dose_max
        
    def optimize(self, objective_func: Callable, initial_dose: float) -> Tuple[np.ndarray, float]:
        """
        执行优化
        
        Args:
            objective_func: 目标函数 f(du_sequence) -> cost
            initial_dose: 当前投矾量
            
        Returns:
            best_du_sequence: 最优控制变化量序列 [Nc]
            best_cost: 最优目标函数值
        """
        raise NotImplementedError


class PSOOptimizer(BaseOptimizer):
    """粒子群优化 (Particle Swarm Optimization)"""
    
    def __init__(self, config: MPCConfig):
        super().__init__(config)
        self.n_particles = config.population_size
        self.max_iter = config.max_iterations
        self.w = config.pso_w      # 惯性权重
        self.c1 = config.pso_c1    # 个体学习因子
        self.c2 = config.pso_c2    # 社会学习因子
        self.rate_limit = config.dose_rate_limit
        
    def optimize(self, objective_func: Callable, initial_dose: float) -> Tuple[np.ndarray, float]:
        dim = self.control_horizon
        
        # 初始化粒子位置（控制变化量）和速度
        # 位置范围: [-rate_limit, rate_limit]
        particles = np.random.uniform(-self.rate_limit, self.rate_limit, 
                                     (self.n_particles, dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, dim))
        
        # 评估初始适应度
        fitness = np.array([objective_func(p, initial_dose) for p in particles])
        
        # 个体最优
        pbest_positions = particles.copy()
        pbest_fitness = fitness.copy()
        
        # 全局最优
        gbest_idx = np.argmin(fitness)
        gbest_position = particles[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        # 迭代优化
        for iteration in range(self.max_iter):
            r1 = np.random.random((self.n_particles, dim))
            r2 = np.random.random((self.n_particles, dim))
            
            # 更新速度
            velocities = (self.w * velocities + 
                         self.c1 * r1 * (pbest_positions - particles) +
                         self.c2 * r2 * (gbest_position - particles))
            
            # 更新位置
            particles = particles + velocities
            
            # 边界处理（控制变化量限制）
            particles = np.clip(particles, -self.rate_limit, self.rate_limit)
            
            # 评估适应度
            fitness = np.array([objective_func(p, initial_dose) for p in particles])
            
            # 更新个体最优
            improved = fitness < pbest_fitness
            pbest_positions[improved] = particles[improved]
            pbest_fitness[improved] = fitness[improved]
            
            # 更新全局最优
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < gbest_fitness:
                gbest_position = particles[min_idx].copy()
                gbest_fitness = fitness[min_idx]
        
        return gbest_position, gbest_fitness


class SAOptimizer(BaseOptimizer):
    """模拟退火 (Simulated Annealing)"""
    
    def __init__(self, config: MPCConfig):
        super().__init__(config)
        self.temp_init = config.sa_temp_init
        self.temp_min = config.sa_temp_min
        self.alpha = config.sa_alpha
        self.max_iter = config.max_iterations
        self.rate_limit = config.dose_rate_limit
        
    def optimize(self, objective_func: Callable, initial_dose: float) -> Tuple[np.ndarray, float]:
        dim = self.control_horizon
        
        # 初始解
        current_solution = np.random.uniform(-self.rate_limit, self.rate_limit, dim)
        current_cost = objective_func(current_solution, initial_dose)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temp = self.temp_init
        
        # 退火过程
        while temp > self.temp_min:
            for _ in range(self.max_iter):
                # 生成邻域解（在当前解附近随机扰动）
                perturbation = np.random.normal(0, 0.5, dim)
                new_solution = current_solution + perturbation
                new_solution = np.clip(new_solution, -self.rate_limit, self.rate_limit)
                
                new_cost = objective_func(new_solution, initial_dose)
                
                # Metropolis 准则
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / temp):
                    current_solution = new_solution
                    current_cost = new_cost
                    
                    # 更新最优解
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost
            
            # 降温
            temp *= self.alpha
        
        return best_solution, best_cost


class DEOptimizer(BaseOptimizer):
    """差分进化 (Differential Evolution)"""
    
    def __init__(self, config: MPCConfig):
        super().__init__(config)
        self.n_population = config.population_size
        self.max_iter = config.max_iterations
        self.F = config.de_f       # 缩放因子
        self.CR = config.de_cr     # 交叉概率
        self.rate_limit = config.dose_rate_limit
        
    def optimize(self, objective_func: Callable, initial_dose: float) -> Tuple[np.ndarray, float]:
        dim = self.control_horizon
        
        # 初始化种群
        population = np.random.uniform(-self.rate_limit, self.rate_limit,
                                      (self.n_population, dim))
        fitness = np.array([objective_func(ind, initial_dose) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # 迭代进化
        for iteration in range(self.max_iter):
            for i in range(self.n_population):
                # 变异：随机选择三个不同的个体
                indices = [idx for idx in range(self.n_population) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # 变异向量
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, -self.rate_limit, self.rate_limit)
                
                # 交叉
                cross_points = np.random.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                    
                trial = np.where(cross_points, mutant, population[i])
                
                # 选择
                trial_fitness = objective_func(trial, initial_dose)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
        
        return best_solution, best_fitness


def create_optimizer(config: MPCConfig) -> BaseOptimizer:
    """
    根据配置创建优化器
    
    Args:
        config: MPC配置
        
    Returns:
        优化器实例
    """
    optimizer_map = {
        'pso': PSOOptimizer,
        'sa': SAOptimizer,
        'de': DEOptimizer,
    }
    
    optimizer_class = optimizer_map.get(config.optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"不支持的优化器类型: {config.optimizer_type}")
    
    return optimizer_class(config)
