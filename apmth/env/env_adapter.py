from gymnasium import Env
from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace, GymEnv
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNReward
from .obs import obs_attrs_to_keep, get_obs_index_map, create_named_tuple, get_graph_features

class Grid2OpEnvAdapter(GymEnv):
    act_attrs_to_keep = ["curtail", "redispatch"]

    def __init__(self, env_name="l2rpn_case14_sandbox"):
        self.base_env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=L2RPNReward)
        super().__init__(self.base_env)
        
        self.observation_space.close()
        self.observation_space = BoxGymObsSpace(self.base_env.observation_space, 
                                                attr_to_keep=obs_attrs_to_keep)
        
        self.action_space.close()
        self.action_space = BoxGymActSpace(self.base_env.action_space, 
                                           attr_to_keep=self.act_attrs_to_keep)
        
        self.index_map = get_obs_index_map(self.base_env)
        self.node_dim = len(obs_attrs_to_keep)

    def get_named_obs(self, obs):
        return create_named_tuple(self.index_map, obs)
    
    def get_obs_graph(self, obs):
        obs = create_named_tuple(self.index_map, obs)
        return get_graph_features(obs, self.base_env)
    
