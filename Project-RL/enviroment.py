import gym
from gym import spaces
import numpy as np
import random

class EnhancedTradingEnv(gym.Env):
    def __init__(self, datasets, initial_balance=10000):
        super(EnhancedTradingEnv, self).__init__()
        self.datasets = datasets
        self.initial_balance = initial_balance

        # Acciones: Hold (0), Buy (1), Sell (2)
        self.action_space = spaces.Discrete(3)

        # Estado: indicadores + balance + posición
        self.state_space = 7  # Modifica según tus características del estado
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(self.state_space,), dtype=np.float32)

        # Variables internas
        self.data = None
        self.current_step = 0
        self.balance = 0
        self.position = 0
        self.max_assets = 0

    def reset(self):
        self.data = random.choice(self.datasets)
        self.n_steps = len(self.data)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.max_assets = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        """Genera el estado basado en datos actuales e indicadores."""
        row = self.data.iloc[self.current_step]
        state = np.array([
            row['Returns_binned'],
            row['SMA_Ratio_binned'],
            row['MACD_hist_binned'],
            row['RSI_binned'],
            row['SO_diff_binned'],
            self.balance,
            self.position
        ], dtype=np.float32)
        return state

    def _calculate_reward(self, action, current_price, previous_assets):
        """Calcula la recompensa basada en la acción, Profit y drawdown."""
        total_assets = self.balance + self.position * current_price
        profit = total_assets - previous_assets
        self.max_assets = max(self.max_assets, total_assets)
        drawdown = (self.max_assets - total_assets) / self.max_assets

        # Penalización por mantener la posición
        hold_penalty = 0.01 if action == 0 else 0

        # Recompensa neta
        reward = profit - drawdown - hold_penalty
        return reward, total_assets

    def step(self, action):
        """Ejecuta la acción y calcula el nuevo estado, recompensa y finalización."""
        current_price = self.data.iloc[self.current_step]["Close"]
        previous_assets = self.balance + self.position * current_price

        # Ejecutar acción
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price

        # Recompensa y nuevo estado
        reward, total_assets = self._calculate_reward(action, current_price, previous_assets)
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        return self._get_observation(), reward, done, {"total_assets": total_assets}

    def render(self, mode="human"):
        """Imprime información del estado actual."""
        current_price = self.data.iloc[self.current_step]["Close"]
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position}, Current Price: {current_price:.2f}")

