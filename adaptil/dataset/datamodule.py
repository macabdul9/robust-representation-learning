import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import VQADataset



class DataModule(pl.LightningDataModule):

    """Lightning Data Module to detach data from model"""

    def __init__(self,
                 path,
                 tokenizer, 
                 img_col="ImageID", 
                 ques_col="Question-En", 
                 ans_col="Answer-En",
                 max_len=16, 
                 transform=None,
                 batch_size=16,
                 num_workers=4,
                 ):

        """
            config: a dicitonary containing data configuration such as batch size, split_size etc
        """
        super().__init__()

        self.path = path
        self.tokenizer = tokenizer
        self.img_col = img_col
        self.ques_col = ques_col
        self.ans_col = ans_col
        self.max_len = max_len
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # prepare and setup the dataset
        self.prepare_data()

    def prepare_data(self):
        """prepare datset"""
        
        self.train_dataset = VQADataset(
            path=self.path,
            set="Train",
            tokenizer=self.tokenizer,
            img_col=self.img_col,
            ques_col=self.ques_col,
            ans_col=self.ans_col
        )
        self.valid_dataset = VQADataset(
            path=self.path,
            set="Valid",
            tokenizer=self.tokenizer,
            img_col=self.img_col,
            ques_col=self.ques_col,
            ans_col=self.ans_col
        )
        self.test_dataset = VQADataset(
            path=self.path,
            set="Test",
            tokenizer=self.tokenizer,
            img_col=self.img_col,
            ques_col=self.ques_col,
            ans_col=self.ans_col
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)