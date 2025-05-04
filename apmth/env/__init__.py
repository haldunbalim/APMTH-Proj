from .obs import *
from .env_adapter import *
from gymnasium import register


register(
    id='l2rpn_case14_sandbox_train-v0',
    entry_point='apmth.env.env_adapter:Grid2OpEnvAdapter',
    kwargs={'env_name': 'l2rpn_case14_sandbox_train'},
)

register(
    id='l2rpn_case14_sandbox_val-v0',
    entry_point='apmth.env.env_adapter:Grid2OpEnvAdapter',
    kwargs={'env_name': 'l2rpn_case14_sandbox_val'},
)

register(
    id='l2rpn_case14_sandbox_test-v0',
    entry_point='apmth.env.env_adapter:Grid2OpEnvAdapter',
    kwargs={'env_name': 'l2rpn_case14_sandbox_test'},
)
