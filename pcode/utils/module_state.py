# -*- coding: utf-8 -*-
import torch
import numpy as np

class ModuleState:
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.keys = set(state_dict.keys())
        self.state_list_sign = None

    def __add__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            new_dict[key] = self.state_dict[key] + other.state_dict[key]
        return ModuleState(new_dict)

    def __iadd__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            self.state_dict[key] += other.state_dict[key]
        return self

    def __sub__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            new_dict[key] = self.state_dict[key] - other.state_dict[key]
        return ModuleState(new_dict)

    def __mul__(self, factor):
        assert isinstance(factor, float) or isinstance(factor, torch.Tensor)
        new_dict = {}
        for key in self.keys:
            data = self.state_dict[key]
            if data.dtype == torch.int64:
                # do nothing for integers
                new_dict[key] = self.state_dict[key]
            else:
                new_dict[key] = self.state_dict[key] * factor
        return ModuleState(new_dict)

    def mul_by_key(self, factor, by_key):
        assert isinstance(factor, float) or isinstance(factor, torch.Tensor)
        new_dict = {}
        for key in self.keys:
            data = self.state_dict[key]
            if data.dtype == torch.int64:
                # do nothing for integers
                new_dict[key] = self.state_dict[key]
            elif by_key is not None and by_key == key:
                new_dict[key] = self.state_dict[key] * factor
            else:
                new_dict[key] = self.state_dict[key]
        return ModuleState(new_dict)

    def __div__(self, factor):
        return self.__mul__(1.0 / factor)

    def gen_grad_sign(self, other_states):
        # walk through self.state_dict and determine the sign
        self.state_list_sign = []
        for k, v in self.state_dict.items():
            self.state_list_sign += [1 if ele > 0 else -1 for ele in (v-other_states[k]).view(-1)]

    def copy_to_module(self, module):
        """
        Use this to copy the state to a module object when you need to maintain the
        computation graph that led to this particular state. This does break the model
        for normal optimizers down the line.
        """
        for name, module in module.named_modules():
            params = module._parameters
            for key in params:
                param_name = f"{name}.{key}"
                if param_name in self.keys:
                    params[key] = self.state_dict[param_name]

    def load_state_to_module(self, module):
        module.load_state_dict(self.state_dict)

    __rmul__ = __mul__
    __truediv__ = __div__
