import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

def _collate_fn(batch):
    default_collate_items = list(batch[0].keys())
    batch_data = []
    for b in batch:
        batch_data.append({k: b[k] for k in default_collate_items})
    data_dict = default_collate(batch_data)
    return data_dict

class DataModuleNoVal(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        train_dataset_name = data_cfg.train.dataset
        self.train_dataset = getattr(import_module(f"src.data.{train_dataset_name.lower()}"), train_dataset_name)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = self.train_dataset(self.data_cfg.train, "train")

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.data_cfg.train.dataloader.batch_size, shuffle=True,
            pin_memory=True, num_workers=self.data_cfg.train.dataloader.num_workers, drop_last=True,
            persistent_workers=(self.data_cfg.train.dataloader.num_workers > 0), collate_fn=_collate_fn
        )

class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        val_data_name = data_cfg.val.dataset
        train_dataset_name = data_cfg.train.dataset
        self.val_dataset = getattr(import_module(f"src.data.{val_data_name.lower()}"), val_data_name)
        self.train_dataset = getattr(import_module(f"src.data.{train_dataset_name.lower()}"), train_dataset_name)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = self.train_dataset(self.data_cfg.train, "train")
            self.val_set = self.val_dataset(self.data_cfg.val, "val")

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.data_cfg.val.dataloader.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.data_cfg.val.dataloader.num_workers, drop_last=False,
            persistent_workers=(self.data_cfg.val.dataloader.num_workers > 0), collate_fn=_collate_fn
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.data_cfg.train.dataloader.batch_size, shuffle=True,
            pin_memory=True, num_workers=self.data_cfg.train.dataloader.num_workers, drop_last=True,
            persistent_workers=(self.data_cfg.train.dataloader.num_workers > 0), collate_fn=_collate_fn
        )
