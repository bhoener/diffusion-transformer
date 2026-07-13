import os
import torch
from src.model import DiT

class EMA:
    def __init__(self, dit: DiT, gammas: tuple[float, ...]=(16.97, 6.94)):
        self.dit = dit
        self.gammas = gammas
        
        self.__reset()
    
    @torch.no_grad()
    def update(self) -> None:
        new_dit_state_dict = self.dit.state_dict()
        for i, gamma in enumerate(self.gammas):
            beta = (1 - 1 / self.step ) ** (gamma + 1)
            
            for k, v in new_dit_state_dict.items():
                self.dit_state_dicts[i][k] = beta * self.dit_state_dicts[i][k] + (1-beta) * v
        self.step += 1
    
    def __reset(self) -> None:
        self.step = 1
        self.dit_state_dicts = [self.dit.state_dict() for _ in range(len(self.gammas))]
        
    def checkpoint(self, root: str, global_step: int, reset: bool = False) -> None:
        if reset:
            for gamma, state_dict in zip(self.gammas, self.dit_state_dicts):
                torch.save(state_dict, os.path.join(root, f"dit_ema_gamma_{gamma:05.2f}_step_{global_step:08d}.pth"))
            self.__reset()
        else:
            for gamma, state_dict in zip(self.gammas, self.dit_state_dicts):
                torch.save(state_dict, os.path.join(root, f"dit_ema_gamma_{gamma:05.2f}.pth"))