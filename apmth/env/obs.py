import numpy as np
import torch
from typing import NamedTuple, Dict, List


gen_feats_keys = ['prod_p', 'prod_v', 'actual_dispatch', 'target_dispatch']
node_feats_keys = ['time_before_cooldown_sub']
load_feats_keys = ['load_p', 'load_q']
obs_attrs_to_keep = list(sorted(gen_feats_keys + node_feats_keys + load_feats_keys))

fields = {key: np.ndarray for key in obs_attrs_to_keep}
Observation = NamedTuple('Observation', fields.items())

def get_graph_features(obs, env):
    obs = Observation(**{k: (v[None] if v.ndim==1 else v) for k, v in zip(obs_attrs_to_keep, obs)})
    bdim = obs.prod_p.shape[0]

    n_node_feats = 0
    if node_feats_keys is not None:
        n_node_feats += len(node_feats_keys)
    if gen_feats_keys is not None:
        n_node_feats += len(gen_feats_keys)
    if load_feats_keys is not None:
        n_node_feats += len(load_feats_keys)

    if isinstance(obs.prod_p, torch.Tensor):
        node_feats = torch.zeros((bdim, len(env.backend.grid_layout), n_node_feats), dtype=torch.float32,
                                 device=obs.prod_p.device)
    else:
        node_feats = np.zeros((bdim, len(env.backend.grid_layout), n_node_feats), dtype=np.float32)

    if gen_feats_keys is not None:
        for i, key in enumerate(gen_feats_keys):
            node_feats[:, env.backend.gen_to_subid, i] = getattr(obs, key)
    if load_feats_keys is not None:
        for i, key in enumerate(load_feats_keys):
            node_feats[:, env.backend.load_to_subid, i +
                       len(gen_feats_keys)] = getattr(obs, key)
    if node_feats_keys is not None:
        for i, key in enumerate(node_feats_keys):
            node_feats[..., i + len(gen_feats_keys) +
                       len(load_feats_keys)] = getattr(obs, key)

    line_beg = env.backend.line_or_to_subid
    line_end = env.backend.line_ex_to_subid
    assert len(line_beg) == len(line_end)
    edge_index = np.repeat(np.vstack((line_beg, line_end))[None], bdim, axis=0)
    edge_index = edge_index.astype(np.int64)
    if isinstance(obs.prod_p, torch.Tensor):
        edge_index = torch.from_numpy(edge_index).to(obs.prod_p.device)

    return node_feats, edge_index


def get_obs_index_map(env):
    n_gen = len(env.backend.gen_to_subid)
    n_node = len(env.backend.grid_layout)
    n_load = len(env.backend.load_to_subid)

    default_obs_attr_to_keep = obs_attrs_to_keep
    key_dim_map = {
        **{k: n_gen for k in gen_feats_keys},
        **{k: n_node for k in node_feats_keys},
        **{k: n_load for k in load_feats_keys},
    }

    index_map = {}
    start = 0
    for key in default_obs_attr_to_keep:
        shape = key_dim_map[key]
        size = shape  # assume 1D per key
        index_map[key] = (start, start + size)
        start += size
    return index_map


def create_named_tuple(index_map: Dict[str, List[int]], vector: np.ndarray):
    # Extract subvectors
    values = {key: vector[..., b:e] for key, (b, e) in index_map.items()}
    return Observation(**values)


if __name__ == "__main__":
    import grid2op
    from grid2op.gym_compat import BoxGymObsSpace, GymEnv
    from lightsim2grid import LightSimBackend

    env_name = "l2rpn_case14_sandbox"
    bk_cls = LightSimBackend
    training_env = grid2op.make(env_name, test=True, backend=bk_cls())

    gym_env = GymEnv(training_env)
    gym_env.observation_space = BoxGymObsSpace(training_env.observation_space,
                                               attr_to_keep=obs_attrs_to_keep)

    obs = training_env.reset()
    obs_gym = gym_env.observation_space.to_gym(obs)

    index_map = get_obs_index_map(training_env)
    ret = create_named_tuple(index_map, obs_gym)

    for k in obs_attrs_to_keep:
        assert np.allclose(getattr(ret, k), getattr(obs, k))
