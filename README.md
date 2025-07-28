# Q-Ctrl-Genesis
# Quantum Control Reinforcement Learning Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.7.0-green?style=flat-square)](https://stable-baselines3.readthedocs.io/en/master/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.0-red?style=flat-square)](https://gymnasium.farama.org/)
[![QuTiP](https://img.shields.io/badge/QuTiP-4.7.0-purple?style=flat-square)](http://qutip.org/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3.2-orange?style=flat-square&logo=hydra)](https://hydra.cc/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14.0-blueviolet?style=flat-square&logo=mlflow)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

`q_ctrl_genesis_v1` is a reinforcement learning toolkit focused on achieving precise quantum control of a single-qubit system. It leverages modern RL algorithms and robust experiment management to explore optimal control pulses.

It features three key attributes:

* **Precise Control:** Utilizes PPO to discover highly effective control pulses for quantum state manipulation.
* **RL-based Adaptation:** Employs model-free reinforcement learning, allowing the agent to adapt to complex quantum dynamics and potential noise without explicit system modeling.
* **Replicable Experiments:** Integrates Hydra for configurable parameters and MLflow for comprehensive experiment tracking, ensuring reproducibility and easy comparison of results.

## Project Structure

The project is organized as follows:

```
q_ctrl_genesis_v1/
├── src/
│   └── quantum_env.py      # Defines the single-qubit quantum environment
├── conf/
│   └── config.yaml         # Hydra configuration file for parameters
├── train.py                # Main training and evaluation script
├── requirements.txt        # Python package dependencies
├── fidelity_and_pulse_plot.png # Example training result plot (fidelity & pulse)
└── ppo_quantum_control_model.zip # Saved trained PPO model
```

## Environment Setup

First, please ensure you have Python 3.10.9 or higher installed.

1.  **Create and Activate a Virtual Environment**

    It is highly recommended to use a virtual environment to manage dependencies:

    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For Linux/macOS:
    source venv/bin/activate
    ```

2.  **Install Dependencies**

    Project dependencies are listed in `requirements.txt`. Install them using pip:

    **`requirements.txt` Content:**
    ```plaintext
    qutip
    stable-baselines3[extra]
    gymnasium
    numpy
    scipy
    matplotlib
    qutip-qip
    mlflow
    hydra-core
    ```

    Installation command:

    ```bash
    pip install -r requirements.txt
    ```

## Core Components

### 1. Quantum Environment (`src/quantum_env.py`)

This file defines the `SingleQubitEnv` class, a custom `gymnasium.Env` that models a single-qubit system. The environment simulates the qubit's time evolution under a drift Hamiltonian and a control Hamiltonian, allowing an RL agent to apply control pulses. It tracks the fidelity of the current quantum state with a predefined target state, and incorporates noise models (dephasing, amplitude damping), slew rate limits, and DAC quantization.

**`src/quantum_env.py` Content:**
```python
import gymnasium as gym
import numpy as np
import qutip as qt
from gymnasium import spaces
import qutip.qip.operations as qip_ops

class SingleQubitEnv(gym.Env):
    """
    A Single Qubit Gate Optimization Environment for Reinforcement Learning.

    Objective: Learn control pulses to evolve an initial state (|0>) to a target state (e.g., |+>).

    - Observation Space: Flattened real and imaginary parts of the density matrix.
    - Action Space: Amplitude of the control pulse applied at each time step.
    Reward:Step-wise reward based on fidelity improvement with the target state.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_steps=50, dt=0.1, gamma_dephasing=0.0, gamma_amplitude_damping=0.0,
                 slew_rate_limit=np.inf, control_noise_std=0.0, dac_bits=np.inf): # Added dac_bits
        super(SingleQubitEnv, self).__init__()

        # Physical System Parameters
        self.n_steps = n_steps  # Total evolution steps
        self.dt = dt            # Time length of each step
        self.times = np.linspace(0, self.n_steps * self.dt, self.n_steps + 1) # Time points
        self.gamma_dephasing = gamma_dephasing       # Dephasing rate
        self.gamma_amplitude_damping = gamma_amplitude_damping # Amplitude damping rate
        self.slew_rate_limit = slew_rate_limit # Max change in control amplitude per second
        self.control_noise_std = control_noise_std # Standard deviation of Gaussian noise on control pulse
        self.dac_bits = dac_bits # Number of bits for DAC quantization

        # Hamiltonian Definition (H = H0 + u(t)*Hc)
        self.H0 = qt.qzero(2)  # Drift Hamiltonian (assumed 0)
        self.Hc = qt.sigmax() # Control Hamiltonian (X-direction)

        # Collapse operators for noise (c_ops)
        self.c_ops = [] # Initialize c_ops as an empty list
        if self.gamma_dephasing > 0:
            self.c_ops.append(np.sqrt(self.gamma_dephasing) * qt.sigmaz())
        if self.gamma_amplitude_damping > 0:
            self.c_ops.append(np.sqrt(self.gamma_amplitude_damping) * qt.destroy(2))

        # State Definitions
        self.initial_state = qt.basis(2, 0)  # Initial state |0>
        self.target_gate = qip_ops.hadamard_transform() # Target: Hadamard gate
        self.target_state = (self.target_gate * self.initial_state).unit() # Target state |+>

        # Gym Environment Interface Definition
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # Control amplitude [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32) # Density matrix (real/imag)

        # Internal State
        self.current_step = 0
        self.current_state = None
        self.last_fidelity = 0.0
        self.history = {'actions': [], 'fidelities': []}
        self.last_control_amplitude = 0.0 # Track previous control amplitude for slew rate limit

    def _get_obs(self):
        """Converts QuTiP quantum state to RL Agent observation."""
        rho = self.current_state
        obs = np.concatenate([rho.full().real.flatten(), rho.full().imag.flatten()]).astype(np.float32)
        return obs

    def _get_info(self):
        """Provides additional debug information (e.g., current fidelity)."""
        fidelity = qt.fidelity(self.current_state, self.target_state)
        return {"fidelity": fidelity}

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_state = qt.ket2dm(self.initial_state) # Initial state as density matrix
        self.last_fidelity = qt.fidelity(self.current_state, self.target_state)
        self.history = {'actions': [], 'fidelities': [self.last_fidelity]}
        self.last_control_amplitude = 0.0 # Reset last control amplitude
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Executes one time step of evolution."""
        desired_control_amplitude = action[0] # Agent's desired control amplitude

        # Apply slew rate limit
        max_change = self.slew_rate_limit * self.dt 
        clipped_change = np.clip(desired_control_amplitude - self.last_control_amplitude, 
                                 -max_change, max_change)
        actual_control_amplitude = self.last_control_amplitude + clipped_change

        # Apply control signal noise
        if self.control_noise_std > 0:
            actual_control_amplitude += np.random.normal(0, self.control_noise_std)
        
        # Apply DAC quantization
        if self.dac_bits != np.inf:
            # Calculate the number of discrete levels
            num_levels = 2**self.dac_bits
            # Scale the amplitude from [-1, 1] to [0, num_levels - 1]
            scaled_amplitude = (actual_control_amplitude + 1.0) / 2.0 * (num_levels - 1)
            # Quantize to the nearest integer level
            quantized_level = np.round(scaled_amplitude)
            # Scale back to [-1, 1] range
            actual_control_amplitude = (quantized_level / (num_levels - 1)) * 2.0 - 1.0

        # Ensure the actual control amplitude stays within the action space bounds [-1, 1]
        actual_control_amplitude = np.clip(actual_control_amplitude, 
                                           self.action_space.low[0], self.action_space.high[0])

        # Update last control amplitude for the next step
        self.last_control_amplitude = actual_control_amplitude

        # Construct time-dependent Hamiltonian
        H = [self.H0, [self.Hc, actual_control_amplitude]] # Use actual_control_amplitude
        tlist = [0, self.dt]

        # Use QuTiP solver for evolution, now including c_ops for noise
        result = qt.mesolve(H, self.current_state, tlist, c_ops=self.c_ops, e_ops=[])
        self.current_state = result.states[-1]

        # Calculate reward (Reward Shaping)
        current_fidelity = qt.fidelity(self.current_state, self.target_state)
        reward = (current_fidelity - self.last_fidelity) * 100 # Scale reward
        
        # Add a large terminal reward on the last step
        if self.current_step == self.n_steps - 1:
            reward += current_fidelity**2 * 10 

        self.last_fidelity = current_fidelity

        # Update internal state and history
        self.current_step += 1
        self.history['actions'].append(actual_control_amplitude) # Log actual amplitude
        self.history['fidelities'].append(current_fidelity)

        # Determine if episode is terminated
        terminated = self.current_step >= self.n_steps
        truncated = False # Not truncated in this scenario

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode='human'):
        """Renders the environment (simple text output)."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Fidelity: {self.last_fidelity:.4f}")
```

### 2. Configuration (`conf/config.yaml`)

This YAML file, managed by Hydra, centralizes all configurable parameters for the environment, training process, model, and MLflow logging. This allows for flexible experimentation and easy parameter tuning via command-line overrides.

**`conf/config.yaml` Content:**
```yaml
# Default configuration for Quantum Control PPO Experiment

# Environment parameters
env:
  n_steps: 50
  dt: 0.2
  gamma_dephasing: 0.0 # Dephasing rate (0.0 for no dephasing)
  gamma_amplitude_damping: 0.0 # Amplitude damping rate (0.0 for no amplitude damping)
  slew_rate_limit: 100.0 # Max change in control amplitude per second (e.g., 100.0 means +/- 100 units/sec). Use np.inf for no limit.
  control_noise_std: 0.0 # Standard deviation of Gaussian noise added to control pulse (0.0 for no noise)
  dac_bits: 8.0 # Number of bits for DAC quantization (default to 8 for example)

# Training parameters
train:
  total_timesteps: 50000

# Model parameters
model:
  policy_type: "MlpPolicy"
  algorithm: "PPO"
  tensorboard_log_dir: "./tensorboard_logs/"

# MLflow parameters
mlflow:
  # This is the naming convention!
  # ${env.dac_bits} will be automatically replaced with the current dac_bits value
  run_name: "DAC_Experiment_Bits_${env.dac_bits}"
```

### 3. Training Script (`train.py`)

The `train.py` script is the entry point for the project. It orchestrates the entire RL pipeline: initializing the quantum environment, setting up the PPO agent, running the training loop, and evaluating the trained model. It also integrates with MLflow to log all relevant experiment data.

**`train.py` Content:**
```python
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.quantum_env import SingleQubitEnv
import mlflow # Import mlflow for logging

import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf # For accessing config parameters
import os # For path operations

def plot_results(env, model):
    """
    Evaluates the trained model once and plots the results.
    Shows the quantum state fidelity evolution over time and the optimized control pulse shape.
    """
    # Reset the environment to start a new evaluation episode
    obs, info = env.reset()
    # Record initial fidelity
    fidelities = [info['fidelity']]
    actions = []
    
    # Perform a full evolution run, for the same number of steps as the environment setting
    for _ in range(env.n_steps):
        # Use the trained model to predict the optimal action
        # deterministic=True ensures no randomness in prediction for a consistent result
        action, _states = model.predict(obs, deterministic=True)
        # Execute the action in the environment to get new state, reward, etc.
        obs, reward, terminated, truncated, info = env.step(action)
        # Record current fidelity
        fidelities.append(info['fidelity'])
        # Record current action (control pulse amplitude)
        actions.append(action[0])
        # If the episode terminates or truncates, break the loop
        if terminated or truncated:
            break

    # Print final fidelity for quick reference
    final_fidelity = fidelities[-1]
    print(f"Final Fidelity after evaluation: {final_fidelity:.5f}")

    # Log final fidelity to MLflow for tracking
    mlflow.log_metric("final_fidelity", final_fidelity)

    # Plotting
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Fidelity over time
    # env.times contains all time points from 0 to the total evolution time
    time_axis = env.times
    ax1.plot(time_axis, fidelities, '.-', label='Fidelity') # '.-' for points and lines
    ax1.set_ylabel('Fidelity')
    ax1.set_ylim([0, 1.05]) # Set Y-axis range slightly larger than [0, 1] for better visualization
    ax1.set_title('Quantum State Fidelity Evolution')
    ax1.grid(True) # Display grid

    # Plot 2: Optimized Control Pulse
    # time_axis[:-1] is used because the number of actions is one less than time points
    # (an action is taken at the end of each time step).
    # where='mid' aligns the step plot centers with time points.
    ax2.step(time_axis[:-1], actions, where='mid', label='Control Pulse Amplitude')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Optimized Control Pulse')
    ax2.grid(True)
    
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

    # Save plot as an artifact in MLflow
    plot_filename = "fidelity_and_pulse_plot.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.close(fig) # Close the plot to prevent it from showing up in subsequent runs if not desired

# Use Hydra decorator to enable configuration management
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print the loaded configuration for verification
    print(OmegaConf.to_yaml(cfg))

    # Set MLflow tracking URI to the local server
    mlflow.set_tracking_uri("[http://127.0.0.1:5000](http://127.0.0.1:5000)")

    # Access parameters from the configuration object (cfg) and ensure correct types
    # These parameters are defined in conf/config.yaml
    n_steps = cfg.env.n_steps
    dt = cfg.env.dt
    gamma_dephasing = cfg.env.gamma_dephasing
    gamma_amplitude_damping = cfg.env.gamma_amplitude_damping
    
    # Convert slew_rate_limit and dac_bits to float to handle 'inf' string correctly
    # Use float() which can parse 'inf' string to float('inf')
    slew_rate_limit = float(cfg.env.slew_rate_limit)
    dac_bits = float(cfg.env.dac_bits) 

    control_noise_std = cfg.env.control_noise_std
    total_timesteps = cfg.train.total_timesteps
    policy_type = cfg.model.policy_type
    algorithm = cfg.model.algorithm
    tensorboard_log_dir = cfg.model.tensorboard_log_dir
    mlflow_run_name = cfg.mlflow.run_name

    # Start an MLflow run to log all experiment details
    with mlflow.start_run(run_name=mlflow_run_name):
        # Log all parameters from Hydra config to MLflow
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # 1. Create the Quantum Environment, passing all parameters
        env = SingleQubitEnv(n_steps=n_steps, dt=dt, 
                             gamma_dephasing=gamma_dephasing, 
                             gamma_amplitude_damping=gamma_amplitude_damping,
                             slew_rate_limit=slew_rate_limit, 
                             control_noise_std=control_noise_std,
                             dac_bits=dac_bits)
        # Optional: Check Gym API compliance. Uncomment for initial environment debugging.
        # check_env(env)

        # 2. Create the PPO Model
        # `policy_type` (e.g., "MlpPolicy") defines the neural network architecture.
        # `env` is the created environment instance.
        # `verbose=1` enables logging of training progress to console.
        # `tensorboard_log` specifies the directory for TensorBoard logs.
        model = PPO(policy_type, env, verbose=1, tensorboard_log=tensorboard_log_dir)

        # 3. Train the Model
        print("Starting training...")
        model.learn(total_timesteps=total_timesteps)
        print("Training finished!")

        # 4. Save the Model
        # The trained model is saved as a zip file, which can be loaded later.
        model_filename = "ppo_quantum_control_model.zip"
        model.save(model_filename)
        mlflow.log_artifact(model_filename) # Log the saved model as an MLflow artifact
        
        # 5. Evaluate and Plot Results
        print("\nEvaluating trained model...")
        plot_results(env, model)

if __name__ == '__main__':
    main()
```

## How to Run

Running the project involves starting the MLflow tracking server and then executing the training script.

1.  **Start the MLflow UI**

    Open a terminal and run the MLflow UI. This will start a local server to track your experiments. Keep this terminal window open.

    ```bash
    mlflow ui
    ```

    Access the MLflow UI in your web browser at: `http://127.0.0.1:5000`

2.  **Run Training**

    Open a **separate** terminal window, navigate to the project root directory (`q_ctrl_genesis_v1/`), activate your virtual environment, and then run the `train.py` script.

    You can use Hydra's powerful command-line override feature to modify parameters defined in `conf/config.yaml` without editing the file directly.

    * **Run with Default Configuration:**
        ```bash
        python train.py
        ```

    * **Adjust Total Training Timesteps:**
        ```bash
        python train.py train.total_timesteps=100000
        ```

    * **Adjust Environment Parameters (e.g., Dephasing Rate):**
        ```bash
        python train.py env.gamma_dephasing=0.01
        ```

    * **Perform a Multi-Run Experiment (e.g., Compare Different DAC Bit Quantization Effects):**
        Hydra's `--multirun` flag allows running multiple experiments with different parameter combinations in one go.

        ```bash
        python train.py env.dac_bits=inf,8,6 --multirun mlflow.run_name="DAC_Bit_Study"
        ```

        **Example Console Output during Training:**
        ```
        Using cpu device
        Wrapping the env with a `Monitor` wrapper
        Wrapping the env in a DummyVecEnv.
        Starting training...
        Logging to ./tensorboard_logs/PPO_1
        ----------------------------------
        | rollout/            |          |
        | time/               |          |
        |    fps              | 123      |
        |    iterations       | 1        |
        |    time_elapsed     | 1        |
        |    total_timesteps  | 2048     |
        | train/              |          |
        |    approx_kl        | 0.000318 |
        |    clip_fraction    | 0        |
        |    ...              |          |
        ----------------------------------
        Training finished!

        Evaluating trained model...
        Final Fidelity after evaluation: 0.70711
        ```

## Results and Analysis

Upon completion of a training run, the `train.py` script will:
* Automatically save the trained PPO model as `ppo_quantum_control_model.zip`.
* Generate a plot (named `fidelity_and_pulse_plot.png`) visualizing the quantum state fidelity evolution over time and the shape of the optimized control pulse.
* All training logs, configured parameters, performance metrics (like final fidelity), and the generated plot will be automatically logged to MLflow. You can use the MLflow UI to easily compare results across different experimental runs, analyze metrics, and download artifacts.

Here are the training result plots you provided, with descriptive captions:

###                      Figure Basic Setup: Quantum State Fidelity Evolution and Optimized Control Pulse (Example Basic Setup)
<img width="1000" height="607" alt="Q-Ctrl Genesis_Figure_Basic Setup" src="https://github.com/user-attachments/assets/e2452083-212e-4390-b79d-35119afe213e" />

###                      Figure 2bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 2 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_2" src="https://github.com/user-attachments/assets/3cd0951a-cc04-4156-a5a7-4f99b557f622" />

###                      Figure 4bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 4 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_4" src="https://github.com/user-attachments/assets/1810fd5b-84f7-4365-80aa-54fa265f4aad" />

###                      Figure 6bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 6 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_4" src="https://github.com/user-attachments/assets/167689d8-bbb3-4ad4-835f-eee8e9ff5496" />

###                      Figure 8bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 8 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_8" src="https://github.com/user-attachments/assets/79dfa3e8-6283-4c12-b933-2d8ddd638bfb" />

###                      Figure 10bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 10 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_10" src="https://github.com/user-attachments/assets/98e9d063-9b07-4aaf-b21b-72fa1e02e721" />

###                      Figure 12bit: Quantum State Fidelity Evolution and Optimized Control Pulse (Example 12 bit)
<img width="1280" height="616" alt="Q-Ctrl Genesis_Figure_12" src="https://github.com/user-attachments/assets/68e35ad0-4526-4f1e-a347-f0e8f3e70313" />

---
"""

# The content of requirements.txt
requirements_txt_content = """
qutip
stable-baselines3[extra]
gymnasium
numpy
scipy
matplotlib
qutip-qip
mlflow
hydra-core
"""

# The content of src/quantum_env.py
quantum_env_py_content = """
import gymnasium as gym
import numpy as np
import qutip as qt
from gymnasium import spaces
import qutip.qip.operations as qip_ops

class SingleQubitEnv(gym.Env):
    \"\"\"
    A Single Qubit Gate Optimization Environment for Reinforcement Learning.

    Objective: Learn control pulses to evolve an initial state (|0>) to a target state (e.g., |+>).

    - Observation Space: Flattened real and imaginary parts of the density matrix.
    - Action Space: Amplitude of the control pulse applied at each time step.
    Reward:Step-wise reward based on fidelity improvement with the target state.
    \"\"\"
    metadata = {\"render_modes\": [\"human\"]}

    def __init__(self, n_steps=50, dt=0.1, gamma_dephasing=0.0, gamma_amplitude_damping=0.0,
                 slew_rate_limit=np.inf, control_noise_std=0.0, dac_bits=np.inf): # Added dac_bits
        super(SingleQubitEnv, self).__init__()

        # Physical System Parameters
        self.n_steps = n_steps  # Total evolution steps
        self.dt = dt            # Time length of each step
        self.times = np.linspace(0, self.n_steps * self.dt, self.n_steps + 1) # Time points
        self.gamma_dephasing = gamma_dephasing       # Dephasing rate
        self.gamma_amplitude_damping = gamma_amplitude_damping # Amplitude damping rate
        self.slew_rate_limit = slew_rate_limit # Max change in control amplitude per second
        self.control_noise_std = control_noise_std # Standard deviation of Gaussian noise on control pulse
        self.dac_bits = dac_bits # Number of bits for DAC quantization

        # Hamiltonian Definition (H = H0 + u(t)*Hc)
        self.H0 = qt.qzero(2)  # Drift Hamiltonian (assumed 0)
        self.Hc = qt.sigmax() # Control Hamiltonian (X-direction)

        # Collapse operators for noise (c_ops)
        self.c_ops = [] # Initialize c_ops as an empty list
        if self.gamma_dephasing > 0:
            self.c_ops.append(np.sqrt(self.gamma_dephasing) * qt.sigmaz())
        if self.gamma_amplitude_damping > 0:
            self.c_ops.append(np.sqrt(self.gamma_amplitude_damping) * qt.destroy(2))

        # State Definitions
        self.initial_state = qt.basis(2, 0)  # Initial state |0>
        self.target_gate = qip_ops.hadamard_transform() # Target: Hadamard gate
        self.target_state = (self.target_gate * self.initial_state).unit() # Target state |+>

        # Gym Environment Interface Definition
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # Control amplitude [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32) # Density matrix (real/imag)

        # Internal State
        self.current_step = 0
        self.current_state = None
        self.last_fidelity = 0.0
        self.history = {'actions': [], 'fidelities': []}
        self.last_control_amplitude = 0.0 # Track previous control amplitude for slew rate limit

    def _get_obs(self):
        \"\"\"Converts QuTiP quantum state to RL Agent observation.\"\"\"
        rho = self.current_state
        obs = np.concatenate([rho.full().real.flatten(), rho.full().imag.flatten()]).astype(np.float32)
        return obs

    def _get_info(self):
        \"\"\"Provides additional debug information (e.g., current fidelity).\"\"\"
        fidelity = qt.fidelity(self.current_state, self.target_state)
        return {\"fidelity\": fidelity}

    def reset(self, seed=None, options=None):
        \"\"\"Resets the environment to its initial state.\"\"\"
        super().reset(seed=seed)
        self.current_step = 0
        self.current_state = qt.ket2dm(self.initial_state) # Initial state as density matrix
        self.last_fidelity = qt.fidelity(self.current_state, self.target_state)
        self.history = {'actions': [], 'fidelities': [self.last_fidelity]}
        self.last_control_amplitude = 0.0 # Reset last control amplitude
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        \"\"\"Executes one time step of evolution.\"\"\"
        desired_control_amplitude = action[0] # Agent's desired control amplitude

        # Apply slew rate limit
        max_change = self.slew_rate_limit * self.dt 
        clipped_change = np.clip(desired_control_amplitude - self.last_control_amplitude, 
                                 -max_change, max_change)
        actual_control_amplitude = self.last_control_amplitude + clipped_change

        # Apply control signal noise
        if self.control_noise_std > 0:
            actual_control_amplitude += np.random.normal(0, self.control_noise_std)
        
        # Apply DAC quantization
        if self.dac_bits != np.inf:
            # Calculate the number of discrete levels
            num_levels = 2**self.dac_bits
            # Scale the amplitude from [-1, 1] to [0, num_levels - 1]
            scaled_amplitude = (actual_control_amplitude + 1.0) / 2.0 * (num_levels - 1)
            # Quantize to the nearest integer level
            quantized_level = np.round(scaled_amplitude)
            # Scale back to [-1, 1] range
            actual_control_amplitude = (quantized_level / (num_levels - 1)) * 2.0 - 1.0

        # Ensure the actual control amplitude stays within the action space bounds [-1, 1]
        actual_control_amplitude = np.clip(actual_control_amplitude, 
                                           self.action_space.low[0], self.action_space.high[0])

        # Update last control amplitude for the next step
        self.last_control_amplitude = actual_control_amplitude

        # Construct time-dependent Hamiltonian
        H = [self.H0, [self.Hc, actual_control_amplitude]] # Use actual_control_amplitude
        tlist = [0, self.dt]

        # Use QuTiP solver for evolution, now including c_ops for noise
        result = qt.mesolve(H, self.current_state, tlist, c_ops=self.c_ops, e_ops=[])
        self.current_state = result.states[-1]

        # Calculate reward (Reward Shaping)
        current_fidelity = qt.fidelity(self.current_state, self.target_state)
        reward = (current_fidelity - self.last_fidelity) * 100 # Scale reward
        
        # Add a large terminal reward on the last step
        if self.current_step == self.n_steps - 1:
            reward += current_fidelity**2 * 10 

        self.last_fidelity = current_fidelity

        # Update internal state and history
        self.current_step += 1
        self.history['actions'].append(actual_control_amplitude) # Log actual amplitude
        self.history['fidelities'].append(current_fidelity)

        # Determine if episode is terminated
        terminated = self.current_step >= self.n_steps
        truncated = False # Not truncated in this scenario

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode='human'):
        \"\"\"Renders the environment (simple text output).\"\"\"
        if mode == 'human':
            print(f\"Step: {self.current_step}, Fidelity: {self.last_fidelity:.4f}\")
"""

# The content of conf/config.yaml
config_yaml_content = """
# Default configuration for Quantum Control PPO Experiment

# Environment parameters
env:
  n_steps: 50
  dt: 0.2
  gamma_dephasing: 0.0 # Dephasing rate (0.0 for no dephasing)
  gamma_amplitude_damping: 0.0 # Amplitude damping rate (0.0 for no amplitude damping)
  slew_rate_limit: 100.0 # Max change in control amplitude per second (e.g., 100.0 means +/- 100 units/sec). Use np.inf for no limit.
  control_noise_std: 0.0 # Standard deviation of Gaussian noise added to control pulse (0.0 for no noise)
  dac_bits: 8.0 # Number of bits for DAC quantization (default to 8 for example)

# Training parameters
train:
  total_timesteps: 50000

# Model parameters
model:
  policy_type: "MlpPolicy"
  algorithm: "PPO"
  tensorboard_log_dir: "./tensorboard_logs/"

# MLflow parameters
mlflow:
  # This is the naming convention!
  # ${env.dac_bits} will be automatically replaced with the current dac_bits value
  run_name: "DAC_Experiment_Bits_${env.dac_bits}"
"""

# The content of train.py
train_py_content = """
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.quantum_env import SingleQubitEnv
import mlflow # Import mlflow for logging

import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf # For accessing config parameters
import os # For path operations

def plot_results(env, model):
    \"\"\"
    Evaluates the trained model once and plots the results.
    Shows the quantum state fidelity evolution over time and the optimized control pulse shape.
    \"\"\"
    # Reset the environment to start a new evaluation episode
    obs, info = env.reset()
    # Record initial fidelity
    fidelities = [info['fidelity']]
    actions = []
    
    # Perform a full evolution run, for the same number of steps as the environment setting
    for _ in range(env.n_steps):
        # Use the trained model to predict the optimal action
        # deterministic=True ensures no randomness in prediction for a consistent result
        action, _states = model.predict(obs, deterministic=True)
        # Execute the action in the environment to get new state, reward, etc.
        obs, reward, terminated, truncated, info = env.step(action)
        # Record current fidelity
        fidelities.append(info['fidelity'])
        # Record current action (control pulse amplitude)
        actions.append(action[0])
        # If the episode terminates or truncates, break the loop
        if terminated or truncated:
            break

    # Print final fidelity for quick reference
    final_fidelity = fidelities[-1]
    print(f"Final Fidelity after evaluation: {final_fidelity:.5f}")

    # Log final fidelity to MLflow for tracking
    mlflow.log_metric("final_fidelity", final_fidelity)

    # Plotting
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Fidelity over time
    # env.times contains all time points from 0 to the total evolution time
    time_axis = env.times
    ax1.plot(time_axis, fidelities, '.-', label='Fidelity') # '.-' for points and lines
    ax1.set_ylabel('Fidelity')
    ax1.set_ylim([0, 1.05]) # Set Y-axis range slightly larger than [0, 1] for better visualization
    ax1.set_title('Quantum State Fidelity Evolution')
    ax1.grid(True) # Display grid

    # Plot 2: Optimized Control Pulse
    # time_axis[:-1] is used because the number of actions is one less than time points
    # (an action is taken at the end of each time step).
    # where='mid' aligns the step plot centers with time points.
    ax2.step(time_axis[:-1], actions, where='mid', label='Control Pulse Amplitude')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Optimized Control Pulse')
    ax2.grid(True)
    
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

    # Save plot as an artifact in MLflow
    plot_filename = "fidelity_and_pulse_plot.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.close(fig) # Close the plot to prevent it from showing up in subsequent runs if not desired

# Use Hydra decorator to enable configuration management
@hydra.main(config_path=\"conf\", config_name=\"config\", version_base=None)
def main(cfg: DictConfig):
    # Print the loaded configuration for verification
    print(OmegaConf.to_yaml(cfg))

    # Set MLflow tracking URI to the local server
    mlflow.set_tracking_uri(\"http://127.0.0.1:5000\")

    # Access parameters from the configuration object (cfg) and ensure correct types
    # These parameters are defined in conf/config.yaml
    n_steps = cfg.env.n_steps
    dt = cfg.env.dt
    gamma_dephasing = cfg.env.gamma_dephasing
    gamma_amplitude_damping = cfg.env.gamma_amplitude_damping
    
    # Convert slew_rate_limit and dac_bits to float to handle 'inf' string correctly
    # Use float() which can parse 'inf' string to float('inf')
    slew_rate_limit = float(cfg.env.slew_rate_limit)
    dac_bits = float(cfg.env.dac_bits) 

    control_noise_std = cfg.env.control_noise_std
    total_timesteps = cfg.train.total_timesteps
    policy_type = cfg.model.policy_type
    algorithm = cfg.model.algorithm
    tensorboard_log_dir = cfg.model.tensorboard_log_dir
    mlflow_run_name = cfg.mlflow.run_name

    # Start an MLflow run to log all experiment details
    with mlflow.start_run(run_name=mlflow_run_name):
        # Log all parameters from Hydra config to MLflow
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # 1. Create the Quantum Environment, passing all parameters
        env = SingleQubitEnv(n_steps=n_steps, dt=dt, 
                             gamma_dephasing=gamma_dephasing, 
                             gamma_amplitude_damping=gamma_amplitude_damping,
                             slew_rate_limit=slew_rate_limit, 
                             control_noise_std=control_noise_std,
                             dac_bits=dac_bits)
        # Optional: Check Gym API compliance. Uncomment for initial environment debugging.
        # check_env(env)

        # 2. Create the PPO Model
        # `policy_type` (e.g., \"MlpPolicy\") defines the neural network architecture.
        # `env` is the created environment instance.
        # `verbose=1` enables logging of training progress to console.
        # `tensorboard_log` specifies the directory for TensorBoard logs.
        model = PPO(policy_type, env, verbose=1, tensorboard_log=tensorboard_log_dir)

        # 3. Train the Model
        print(\"Starting training...\")
        model.learn(total_timesteps=total_timesteps)
        print(\"Training finished!\")

        # 4. Save the Model
        # The trained model is saved as a zip file, which can be loaded later.
        model_filename = \"ppo_quantum_control_model.zip\"
        model.save(model_filename)
        mlflow.log_artifact(model_filename) # Log the saved model as an MLflow artifact
        
        # 5. Evaluate and Plot Results
        print(\"\\nEvaluating trained model...\")
        plot_results(env, model)

if __name__ == '__main__':
    main()
"""

# The paths to the image files for the results section
image_paths = {
    "Figure 1": "Q-Ctrl Genesis_Figure_2.png",
    "Figure 2": "Q-Ctrl Genesis_Figure_4.png",
    "Figure 3": "Q-Ctrl Genesis_Figure_6.png",
    "Figure 4": "Q-Ctrl Genesis_Figure_8.png",
    "Figure 5": "Q-Ctrl Genesis_Figure_10.png",
    "Figure 6": "Q-Ctrl Genesis_Figure_12.png",
    "Figure 7": "Q-Ctrl Genesis_Figure_基礎.png"
}

if __name__ == "__main__":
    # This block allows you to print the content of each file.
    # You can run this Python file to get the content of the other files.

    print("--- Content of requirements.txt ---")
    print(requirements_txt_content)

    print("\n--- Content of src/quantum_env.py ---")
    print(quantum_env_py_content)

    print("\n--- Content of conf/config.yaml ---")
    print(config_yaml_content)

    print("\n--- Content of train.py ---")
    print(train_py_content)

    print("\n--- Image Paths for Results ---")
    for fig_name, path in image_paths.items():
        print(f"{fig_name}: ") # Display image references in a readable format
