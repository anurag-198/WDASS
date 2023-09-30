import torch

def get_module(module):
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    return module

class ema_cls() :
    def __init__(self, net, args) :
        super(ema_cls, self).__init__()
        self.ema_model = net
        self.alpha = args.alpha
        self.buffer_keys = None
        self.init_ema_weights(net)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_model(self, net) :
        return get_module(net)

    def _init_ema_weights(self, net, bn_buffer):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model(net).parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        
        
    def _update_ema(self, net, itern, not_ema, bn_buffer):
        #print(self.alpha)
        alpha_teacher = min(1 - 1 / (itern + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model(net).parameters()):
  
            if not param.data.shape:  # scalar tensor
                if not not_ema :
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else :
                    ema_param.data = param.data
            else:
                if not not_ema :
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]
                else : 
                    ema_param.data[:] = param[:].data[:]

class ema_cls_2() :
    def __init__(self, net, args) :
        super(ema_cls_2, self).__init__()
        self.ema_model = net
        self.alpha = args.alpha
        self.buffer_keys = None
        self._init_ema_weights(net)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_model(self, net) :
        return get_module(net)

    def _init_ema_weights(self, net, bn_buffer=False):
        for param in self.get_ema_model().parameters():
            param.detach_()
        state_dict_main = self.get_model(net).state_dict()
        state_dict_ema = self.get_ema_model().state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"


    def _update_ema(self, net, itern, not_ema, bn_buffer):

        alpha_teacher = min(1 - 1 / (itern + 1), self.alpha)
        state_dict_main = self.get_model(net).state_dict()
        state_dict_ema = self.get_ema_model().state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema or not_ema : 
                v_ema.copy_(v_main.clone().detach_())
            else:
                v_ema.copy_(v_ema * self.alpha + (1. - self.alpha) * v_main.clone().detach_())