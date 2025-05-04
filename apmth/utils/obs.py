import numpy as np
import torch

global_feats_keys = ['day_of_week', 'hour_of_day', 'minute_of_hour']
gen_feats_keys = ['prod_p', 'prod_v', 'actual_dispatch', 'target_dispatch']
node_feats_keys = ['time_before_cooldown_sub']
load_feats_keys = ['load_p', 'load_q']
edge_feats_keys = ['time_before_cooldown_line', 'rho', 'timestep_overflow', 'line_status']


def get_node_features(obs, env):
    if global_feats_keys is not None:
        global_feats = np.zeros((len(global_feats_keys),), dtype=np.float32)
        for i, key in enumerate(global_feats_keys):
            global_feats[i] = getattr(obs, key)
    else:
        global_feats = None

    n_node_feats = 0
    if node_feats_keys is not None:
        n_node_feats += len(node_feats_keys)
    if gen_feats_keys is not None:
        n_node_feats += len(gen_feats_keys)
    if load_feats_keys is not None:
        n_node_feats += len(load_feats_keys)
    node_feats = np.zeros((len(env.backend.grid_layout),
                          n_node_feats), dtype=np.float32)

    if gen_feats_keys is not None:
        for i, key in enumerate(gen_feats_keys):
            node_feats[env.backend.gen_to_subid, i] = getattr(obs, key)
    if load_feats_keys is not None:
        for i, key in enumerate(load_feats_keys):
            node_feats[env.backend.load_to_subid, i +
                       len(gen_feats_keys)] = getattr(obs, key)
    if node_feats_keys is not None:
        for i, key in enumerate(node_feats_keys):
            node_feats[:, i + len(gen_feats_keys) +
                       len(load_feats_keys)] = getattr(obs, key)

    line_beg = env.backend.line_or_to_subid
    line_end = env.backend.line_ex_to_subid
    assert len(line_beg) == len(line_end)
    edge_index = np.vstack((line_beg, line_end))

    edge_feats = np.zeros((len(line_beg), len(edge_feats_keys)), dtype=np.float32)

    for i, edge_feat in enumerate(edge_feats_keys):
        edge_feats[:, i] = getattr(obs, edge_feat)

    return global_feats, node_feats, edge_feats, edge_index
