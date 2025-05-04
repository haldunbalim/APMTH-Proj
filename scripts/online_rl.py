import os
import hydra
from omegaconf import DictConfig
import gymnasium as gym
from apmth import *
import torch
from hydra.utils import get_class
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
import wandb
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@hydra.main(config_path="../config/online", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    if cfg.algo == "ppo":
        algo_cls = PPO
    elif cfg.algo == "sac":
        algo_cls = SAC
    else:
        raise ValueError(
            f"Algorithm {cfg.algo} not supported. Use 'ppo' or 'sac'.")

    env = DummyVecEnv([
        lambda: Monitor(gym.make(cfg.env_name + '_train-v0'))
        for _ in range(cfg.train_envs)
    ])
    eval_env = Monitor(gym.make(cfg.env_name + '_val-v0'))


    policy_kwargs = {"net_arch": list(cfg.mlp_layers)}
    if hasattr(cfg, "feature_extractor"):
        feat_args = {
            "features_extractor_class": get_class(cfg.feature_extractor._target_),
            "features_extractor_kwargs": {'env': env.envs[0].unwrapped,
                                          **{k: v for k, v in cfg.feature_extractor.items() if k not in {"_target_", "share_weights"}}},
            "share_features_extractor": cfg.feature_extractor.share_weights,
        }
        policy_kwargs.update(feat_args)
    
    wandb.init(project=cfg.wandb_project, sync_tensorboard=True, dir='.', config=cfg_dict)
    model = algo_cls("MlpPolicy", env, verbose=1, tensorboard_log='.', policy_kwargs=policy_kwargs)
    new_logger = configure('.', ["tensorboard", "stdout"])
    model.set_logger(new_logger)

    eval_callback = EvalCallback(
        eval_env,
        log_path='checkpoints',
        n_eval_episodes=cfg.n_eval,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path='checkpoints',
        name_prefix=cfg.algo,
    )

    callback = CallbackList([eval_callback, checkpoint_callback])
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    model.save(os.path.join('checkpoints', f"final_model"))


if __name__ == "__main__":
    main()