"""
Zero-Shot Strategy Inversion
Pillar 2: Policy-guided search for synthetic unlearning scenarios
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import gymnasium as gym


class StrategyInversion:
    """Zero-shot strategy inversion without world models."""
    
    def __init__(
        self,
        agent,
        env: gym.Env,
        num_seeds: int = 100,
        num_iterations: int = 50,
        learning_rate: float = 0.01,
        search_method: str = "gradient_based",
        temperature: float = 1.0,
        noise_scale: float = 0.1,
        device: str = "cpu",
    ):
        self.agent = agent
        self.env = env
        self.num_seeds = num_seeds
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.search_method = search_method
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.device = torch.device(device)
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def generate_forget_states(
        self,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> List[np.ndarray]:
        """Generate synthetic states where policy should unlearn."""
        forget_states = []
        
        for _ in range(self.num_seeds):
            # Initialize random seed state
            seed_state = self._sample_state()
            
            # Optimize state to match target criteria
            optimized_state = self._optimize_state(
                seed_state,
                target_actions,
                target_value_range,
            )
            
            if optimized_state is not None:
                forget_states.append(optimized_state)
        
        return forget_states
    
    def _sample_state(self) -> np.ndarray:
        """Sample random state from observation space."""
        if isinstance(self.observation_space, gym.spaces.Box):
            return np.random.uniform(
                self.observation_space.low,
                self.observation_space.high,
            )
        else:
            raise NotImplementedError(f"Observation space {type(self.observation_space)} not supported")
    
    def _optimize_state(
        self,
        initial_state: np.ndarray,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> np.ndarray:
        """Optimize state using gradient-based search."""
        if self.search_method == "gradient_based":
            return self._gradient_based_search(initial_state, target_actions, target_value_range)
        elif self.search_method == "random":
            return self._random_search(initial_state, target_actions, target_value_range)
        elif self.search_method == "evolutionary":
            return self._evolutionary_search(initial_state, target_actions, target_value_range)
        else:
            return initial_state
    
    def _gradient_based_search(
        self,
        initial_state: np.ndarray,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> np.ndarray:
        """Gradient-based optimization to push policy toward target region."""
        state = torch.FloatTensor(initial_state).to(self.device)
        state.requires_grad = True
        
        optimizer = torch.optim.Adam([state], lr=self.learning_rate)
        
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Get policy outputs
            with torch.enable_grad():
                action_output, value = self.agent.network(state.unsqueeze(0))
            
            # Compute loss based on target criteria
            loss = torch.tensor(0.0, device=self.device)
            
            if target_actions is not None:
                # Push policy toward specific actions
                if self.agent.action_type == "discrete":
                    target_action = torch.LongTensor([target_actions[0]]).to(self.device)
                    loss += -torch.log_softmax(action_output, dim=-1)[0, target_action]
                else:
                    target_action = torch.FloatTensor([target_actions[0]]).to(self.device)
                    loss += torch.mean((action_output - target_action) ** 2)
            
            if target_value_range is not None:
                # Push value estimate toward target range
                target_value = (target_value_range[0] + target_value_range[1]) / 2
                loss += (value - target_value) ** 2
            
            # Add diversity/exploration term
            loss += -self.noise_scale * torch.randn_like(loss)
            
            loss.backward()
            optimizer.step()
            
            # Clip to observation space bounds
            if isinstance(self.observation_space, gym.spaces.Box):
                with torch.no_grad():
                    state.clamp_(
                        torch.FloatTensor(self.observation_space.low).to(self.device),
                        torch.FloatTensor(self.observation_space.high).to(self.device),
                    )
        
        return state.detach().cpu().numpy()
    
    def _random_search(
        self,
        initial_state: np.ndarray,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> np.ndarray:
        """Random search for states matching criteria."""
        best_state = initial_state
        best_score = float('-inf')
        
        for _ in range(self.num_iterations):
            # Sample random perturbation
            candidate = initial_state + np.random.randn(*initial_state.shape) * self.noise_scale
            
            # Clip to bounds
            if isinstance(self.observation_space, gym.spaces.Box):
                candidate = np.clip(candidate, self.observation_space.low, self.observation_space.high)
            
            # Evaluate
            score = self._evaluate_state(candidate, target_actions, target_value_range)
            
            if score > best_score:
                best_score = score
                best_state = candidate
        
        return best_state
    
    def _evolutionary_search(
        self,
        initial_state: np.ndarray,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> np.ndarray:
        """Evolutionary search with population."""
        population_size = 20
        population = [initial_state + np.random.randn(*initial_state.shape) * self.noise_scale 
                     for _ in range(population_size)]
        
        for _ in range(self.num_iterations):
            # Evaluate population
            scores = [self._evaluate_state(state, target_actions, target_value_range) 
                     for state in population]
            
            # Select top performers
            top_indices = np.argsort(scores)[-population_size//2:]
            survivors = [population[i] for i in top_indices]
            
            # Generate offspring
            offspring = []
            for _ in range(population_size - len(survivors)):
                parent1, parent2 = np.random.choice(survivors, size=2, replace=False)
                child = (parent1 + parent2) / 2 + np.random.randn(*initial_state.shape) * self.noise_scale
                
                # Clip to bounds
                if isinstance(self.observation_space, gym.spaces.Box):
                    child = np.clip(child, self.observation_space.low, self.observation_space.high)
                
                offspring.append(child)
            
            population = survivors + offspring
        
        # Return best from final population
        scores = [self._evaluate_state(state, target_actions, target_value_range) 
                 for state in population]
        best_idx = np.argmax(scores)
        return population[best_idx]
    
    def _evaluate_state(
        self,
        state: np.ndarray,
        target_actions: List = None,
        target_value_range: Tuple[float, float] = None,
    ) -> float:
        """Evaluate how well state matches target criteria."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_output, value = self.agent.network(state_tensor)
        
        score = 0.0
        
        if target_actions is not None:
            if self.agent.action_type == "discrete":
                probs = torch.softmax(action_output, dim=-1)
                score += probs[0, target_actions[0]].item()
            else:
                distance = torch.mean((action_output[0] - torch.FloatTensor(target_actions[0]).to(self.device)) ** 2)
                score -= distance.item()
        
        if target_value_range is not None:
            v = value.item()
            if target_value_range[0] <= v <= target_value_range[1]:
                score += 1.0
            else:
                score -= min(abs(v - target_value_range[0]), abs(v - target_value_range[1]))
        
        return score
