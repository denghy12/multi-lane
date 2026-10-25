"""Explicitly scoped Test intervention for the completed seed1 b32 experiment."""
from .historical_b32_modules import validate_config

TASKS = (0, 2, 7)
ARMS = ('ATTENTION', 'ATTENTION_BOUND', 'POSITION', 'POSITION_CALIBRATED')


def validate_request(config, split, checkpoint_tasks, disabled, expected_tasks,
                     allow_smoke=False, full_cohort=False, disable_moe=False,
                     disable_specialist=False, disable_joint=False):
    arm = validate_config(config)
    expected = 'B' if arm.startswith('ATTENTION') else 'C'
    if (arm not in ARMS or split != 'test' or expected_tasks != 8
            or tuple(checkpoint_tasks or ()) != TASKS
            or tuple(disabled) not in ((), (expected,))
            or allow_smoke or full_cohort or disable_moe or disable_specialist or disable_joint
            or config.get('max_tasks') != 8 or config.get('reporting_split') != 'test'):
        raise ValueError('Historical b32 Test attribution requires the four completed seed1 arms, tasks 0/2/7, and only their own module bypass')
    return arm
