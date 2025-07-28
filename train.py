import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.quantum_env import SingleQubitEnv
import mlflow

import hydra
from omegaconf import DictConfig, OmegaConf
import os 

def plot_results(env, model):
    """
    Evaluates the trained model and plots results.
    Shows fidelity evolution and optimized control pulse.
    """
    obs, info = env.reset() #Reset environment
    fidelities = [info['fidelity']] #Record initial fidelity
    actions = [] # Initialize actions as an empty list
    
    #Perform a full evolution run
    for _ in range(env.n_steps):
        action, _states = model.predict(obs, deterministic=True) #Predict optimal action
        obs, reward, terminated, truncated, info = env.step(action) #Execute action
        fidelities.append(info['fidelity']) #Record current fidelity
        actions.append(action[0]) # Store the scalar value for plotting
        if terminated or truncated:
            break

    final_fidelity = fidelities[-1] # Get final fidelity
    print(f"Final Fidelity after evaluation: {final_fidelity:.5f}")

    # Log final fidelity to MLflow
    mlflow.log_metric("final_fidelity", final_fidelity)

    #Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True) #Create subplots
    
    #Plot 1:Fidelity over time
    time_axis = env.times
    ax1.plot(time_axis, fidelities, '.-', label='Fidelity')
    ax1.set_ylabel('Fidelity')
    ax1.set_ylim([0, 1.05])
    ax1.set_title('Quantum State Fidelity Evolution')
    ax1.grid(True)

    #Plot 2:Optimized Control Pulse
    ax2.step(time_axis[:-1], actions, where='mid', label='Control Pulse Amplitude')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Optimized Control Pulse')
    ax2.grid(True)
    
    plt.tight_layout() #Adjust subplot parameters
    plt.show() #Display plot

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
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Access parameters from the configuration object (cfg) and ensure correct types
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

    # Start an MLflow run
    with mlflow.start_run(run_name=mlflow_run_name):
        # Log parameters from Hydra config
        mlflow.log_param("n_steps", n_steps)
        mlflow.log_param("dt", dt)
        mlflow.log_param("gamma_dephasing", gamma_dephasing)
        mlflow.log_param("gamma_amplitude_damping", gamma_amplitude_damping)
        mlflow.log_param("slew_rate_limit", slew_rate_limit)
        mlflow.log_param("control_noise_std", control_noise_std)
        mlflow.log_param("dac_bits", dac_bits)
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("policy_type", policy_type)
        mlflow.log_param("algorithm", algorithm)

        #1.Create the Quantum Environment, passing all parameters
        env = SingleQubitEnv(n_steps=n_steps, dt=dt, 
                             gamma_dephasing=gamma_dephasing, 
                             gamma_amplitude_damping=gamma_amplitude_damping,
                             slew_rate_limit=slew_rate_limit, 
                             control_noise_std=control_noise_std,
                             dac_bits=dac_bits)
        #check_env(env) #Optional:Check Gym API compliance

        #2.Create the PPO Model
        model = PPO(policy_type, env, verbose=1, tensorboard_log=tensorboard_log_dir)

        #3.Train the Model
        print("Starting training...")
        model.learn(total_timesteps=total_timesteps)
        print("Training finished!")

        #4.Save the Model
        model_filename = "ppo_quantum_control_model.zip"
        model.save(model_filename)
        mlflow.log_artifact(model_filename)
        
        #5.Evaluate and Plot Results
        print("\nEvaluating trained model...")
        plot_results(env, model)

if __name__ == '__main__':
    main()