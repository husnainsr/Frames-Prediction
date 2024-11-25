from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    @abstractmethod
    def forward(self, x, future_frames):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__