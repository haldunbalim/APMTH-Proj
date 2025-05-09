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
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

def load_model(cfg):
    tmp_env = gym.make(cfg.env_name + '_train-v0')
    cls = get_class(cfg.model._target_)
    assert hasattr(cfg, 'ckpt_file'), "Checkpoint directory must be provided"
    ckpt_file = cfg.ckpt_file

    fpath = os.path.abspath(__file__)
    dpath = os.path.abspath(os.path.join(fpath, "../.."))
    ckpt_file = os.path.join(dpath, ckpt_file)
    kwargs = OmegaConf.to_container(
        cfg.model, resolve=True, throw_on_missing=True)
    kwargs = {k: v for k, v in kwargs.items() if k not in ['__target__', 'ckpt_file']}
    ae = cls.load_from_checkpoint(ckpt_file, env=tmp_env.unwrapped, **kwargs)
    return ae


@hydra.main(config_path="../config/offline", config_name="config-rl", version_base="1.1")
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
    ae = load_model(cfg)

    policy_kwargs = {"net_arch": list(cfg.mlp_layers)}
    feat_args = {
        "features_extractor_class": FrozenFeatureExtractor,
        "features_extractor_kwargs": {'encode_fn': ae.encode,
                                      'encode_dim': ae.encode_dim},
        "share_features_extractor": True,
    }
    policy_kwargs.update(feat_args)

    wandb.init(project=cfg.wandb_project,
               sync_tensorboard=True, dir='.', config=cfg_dict)
    wandb.config.update({'log_dir': HydraConfig.get().run.dir,
                        'script_name': os.path.basename(__file__)})
    model = algo_cls("MlpPolicy", env, verbose=1,
                     tensorboard_log='.', policy_kwargs=policy_kwargs)
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
    