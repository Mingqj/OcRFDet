from mmcv.runner import HOOKS, Hook
# from mmcv.runner import LrUpdaterHook

__all__ = ['CustomLrUpdaterHook']

@HOOKS.register_module()
# class CustomLrUpdaterHook(LrUpdaterHook):
class CustomLrUpdaterHook(Hook):
    def __init__(self, layers, step=5, gamma=0.5):
        self.layers = layers
        self.step = step
        self.gamma = gamma

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch % self.step == 0 and epoch != 0:
            for param_group in runner.optimizer.param_groups:
                if 'name' in param_group:
                    for layer in self.layers:
                        if layer in param_group['name']:
                            old_lr = param_group['lr']
                            param_group['lr'] *= self.gamma / (2 ** epoch)
                            runner.logger.info(f"Reduced learning rate of {layer} from {old_lr} to {param_group['lr']} at epoch {epoch}")

        # if epoch % self.step == 0 and epoch != 0:
            # for param_group in runner.optimizer.param_groups:
            #     if 'name' in param_group:
            #         if param_group['name'] in self.layers:
            #             old_lr = param_group['lr']
            #             param_group['lr'] *= self.gamma
            #             runner.logger.info(f"Reduced learning rate of {param_group['name']} from {old_lr} to {param_group['lr']} at epoch {epoch}")
            # print(runner.optimizer.param_groups)
            # for item in range(len(runner.optimizer.param_groups)):
            #     if 'name' in runner.optimizer.param_groups[item]:
            #         if runner.optimizer.param_groups[item]['name'] in self.layers:
            #             old_lr = runner.optimizer.param_groups[item]['lr']
            #             runner.optimizer.param_groups[item]['lr'] = old_lr * self.gamma
            #             runner.logger.info(f"Reduced learning rate of {runner.optimizer.param_groups[item]['name']} from {old_lr} to {runner.optimizer.param_groups[item]['lr']} at epoch {epoch} 222")
        
        # # Log learning rates after potential update for debugging
        # for param_group in runner.optimizer.param_groups:
        #     runner.logger.info(f"Epoch {epoch}: Updated learning rate for {param_group.get('name', 'default')}: {param_group['lr']}")
    
    def after_train_epoch(self, runner):
        epoch = runner.epoch
        # for param_group in runner.optimizer.param_groups:
        #     runner.logger.info(f"Epoch {epoch}: Updated learning rate for {param_group.get('name', 'default')}: {param_group['lr']}")
