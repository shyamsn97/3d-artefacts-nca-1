import torch
import attr
from einops import rearrange

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer

@attr.s(repr=False)
class ReplicationNCATrainer(VoxelCATrainer):

    n_duplications: int = attr.ib(default=5)
    steps_per_duplication: int = attr.ib(default=8)
    norm_grad: bool = attr.ib(default=False)

    def duplicate(self, x, shape=None):
        if shape is None:
            shape = x.shape
        d, h, w = shape[2], shape[3], shape[4]
        x = torch.repeat_interleave(torch.repeat_interleave(torch.repeat_interleave(x, 2, dim=2), 2, dim=3), 2, dim=4)  # cell division
        x = x[
                :,
                :,
                d // 2:(d // 2) + d,
                h // 2:(h // 2) + h,
                w // 2:(w // 2) + w
            ] # cut out middle
        return x

    def train_func(self, x, targets, steps=None):
        self.optimizer.zero_grad()
        x = self.model(x, steps=self.steps_per_duplication, rearrange_output=False)
        shape = x.shape
        for i in range(self.n_duplications):
            x = self.duplicate(x, shape)
            x = self.model(x, steps=self.steps_per_duplication, rearrange_input=False, rearrange_output=False)

        loss, iou_loss = self.get_loss(x, targets)
        loss.backward()

        if self.norm_grad:
            for p in self.model.parameters():
                p.grad /= torch.norm(p.grad) + 1e-8

        self.optimizer.step()
        self.scheduler.step()

        x = rearrange(x, "b c d h w -> b d h w c")

        out = {
            "out": x,
            "metrics": {"loss": loss.item(), "iou_loss": iou_loss.item()},
            "loss": loss,
        }
        return out
