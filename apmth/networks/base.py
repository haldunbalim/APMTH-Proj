import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
import numpy as np


class BaseModel(pl.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()

    def compute_losses(self, input_dict, intermediate_dict):
        raise Exception('Base class')

    def forward(self, input_dict, batch_idx=None, return_intermediate_dict=False):
        intermediate_dict = self.predict_step(input_dict, batch_idx)
        output_dict = self.compute_losses(input_dict, intermediate_dict)
        return (intermediate_dict, output_dict) if return_intermediate_dict else output_dict

    def log_output_dct(self, output_dict, typ):
        for k in output_dict:
            if "loss" in k or "metric" in k:
                self.log(typ+"_"+k, output_dict[k], on_step=False, on_epoch=True,
                         prog_bar=True, logger=True)

    def training_step(self, input_dict, batch_idx):
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "train")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_dict": intermediate_dict,
                "output_dict": output_dict}

    def validation_step(self, input_dict, batch_idx):
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "val")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_dict": intermediate_dict,
                "output_dict": output_dict}

    def test_step(self, input_dict, batch_idx):
        output_dict = self(input_dict, batch_idx=batch_idx)
        self.log_output_dct(output_dict, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
    
    def on_epoch_end(self):
        self.log("epoch", self.current_epoch)