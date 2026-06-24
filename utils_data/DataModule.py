import lightning as L
from torch.utils.data import DataLoader

from utils_data.CoToGraspDataset import CoToGraspDataset

class DataModule(L.LightningDataModule):
    def __init__(self, cfg, shuffle=False):
        super().__init__()
        self.cfg = cfg
        self.shuffle = shuffle
    
    def setup(self, stage=''):
        print("[DataModule] Loading dataset... ")
        self.dataset = CoToGraspDataset(robot_name=self.cfg['robot_name'], fixed_rt=self.cfg['fixed_rt'], num_handprints=None)
        print("[DataModule] Dataset Loaded ! Size  :", self.dataset.__len__())
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.cfg['batch_size'], 
            shuffle=self.shuffle, 
            num_workers=self.cfg['num_workers'], 
            multiprocessing_context='fork' if self.cfg['num_workers'] > 0 else None, 
            persistent_workers=True if self.cfg['num_workers'] > 0 else False)

    
