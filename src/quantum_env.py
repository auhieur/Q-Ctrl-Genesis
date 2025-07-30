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