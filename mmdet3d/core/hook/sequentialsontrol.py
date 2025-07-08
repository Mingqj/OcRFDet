# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    # def __init__(self, temporal_start_epoch=1, layers=None, step=5, gamma=0.5):
    def __init__(self, temporal_start_epoch=1):

        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch
        # self.layers = layers
        # self.step = step
        # self.gamma = gamma

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.with_prev = flag
        elif is_parallel(runner.model):
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        # epoch = runner.epoch
        # # if epoch % self.step == 0 and epoch != 0:
        # for item in range(len(runner.optimizer.param_groups)):
        #         if 'name' in runner.optimizer.param_groups[item]:
        #             if runner.optimizer.param_groups[item]['name'] in self.layers:
        #                 old_lr = runner.optimizer.param_groups[item]['lr']
        #                 runner.optimizer.param_groups[item]['lr'] = old_lr * self.gamma
        #                 runner.logger.info(f"Reduced learning rate of {runner.optimizer.param_groups[item]['name']} from {old_lr} to {runner.optimizer.param_groups[item]['lr']} at epoch {epoch} 111")
        # print(runner.epoch, self.temporal_start_epoch, 1111111111111111111111111111111111)
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)