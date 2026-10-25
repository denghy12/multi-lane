import unittest
from multi_lane.track_a.historical_b32_modules import expected
from multi_lane.track_a.historical_b32_test_attribution import ARMS, validate_request


def config(arm):
    c = expected(arm)
    c.update(eval_workers=2, learning_rate=.0125, adapter_learning_rate=.0004,
        view_fusion_learning_rate=None if arm == 'BASE' else .0004,
        parameter_group_loss_routing='adapter_asl', adapter_parameter_objective='asl',
        optimizer='Adam_reset_per_task', adapter_task_initialization='independent',
        amp=True, tf32=True, fixed_view_sampler_protocol=None,
        evaluation_scope='samples_intersect_seen_classes', max_tasks=8, reporting_split='test')
    return c


class HistoricalB32TestAttributionTest(unittest.TestCase):
    def test_four_on_and_own_module_off_requests(self):
        for arm in ARMS:
            for disabled in ((), ('B' if arm.startswith('ATTENTION') else 'C',)):
                self.assertEqual(validate_request(config(arm), 'test', [0,2,7], disabled, 8), arm)

    def test_wrong_module_or_adapter_bypass_is_rejected(self):
        for arm in ARMS:
            for disabled in (('A',), ('B','C'), ('C' if arm.startswith('ATTENTION') else 'B',)):
                with self.assertRaises(ValueError):validate_request(config(arm),'test',[0,2,7],disabled,8)

    def test_other_boundaries_and_incomplete_sources_are_rejected(self):
        for tasks in (None, [0], [7], [0,2,6], [7,2,0], [0,2,7,7]):
            with self.assertRaises(ValueError):validate_request(config('ATTENTION'),'test',tasks,('B',),8)
        with self.assertRaises(ValueError):validate_request(config('ATTENTION'),'test',[0,2,7],('B',),2)
        c=config('ATTENTION');c['max_tasks']=2
        with self.assertRaises(ValueError):validate_request(c,'test',[0,2,7],('B',),8)

    def test_cohort_smoke_and_unrelated_interventions_are_rejected(self):
        for key in ('allow_smoke','full_cohort','disable_moe','disable_specialist','disable_joint'):
            with self.assertRaises(ValueError):validate_request(config('POSITION'),'test',[0,2,7],('C',),8,**{key:True})
        with self.assertRaises(ValueError):validate_request(config('POSITION'),'val',[0,2,7],('C',),8)

    def test_profile_and_historical_numerics_remain_locked(self):
        with self.assertRaises(ValueError):validate_request(config('BASE'),'test',[0,2,7],(),8)
        for key,value in (('seed',0),('adapter_residual_scale',.1),('view_auxiliary_loss_weight',0.),('historical_b32_modules',False)):
            c=config('POSITION');c[key]=value
            with self.assertRaises(ValueError):validate_request(c,'test',[0,2,7],('C',),8)


if __name__ == '__main__':unittest.main()
