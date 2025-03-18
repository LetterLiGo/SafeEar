import math
import torch
import pytorch_lightning as pl
from ..losses.loss import compute_eer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()

class SafeEarTrainer(pl.LightningModule):
    def __init__(
            self,
            decouple_model,
            detect_model,
            lr_raw_former,
            save_score_path
        ) -> None:
        super().__init__()
        
        self.decouple_model = decouple_model
        self.detect_model = detect_model
        self.lr_raw_former = lr_raw_former
        self.save_score_path = save_score_path 
            
        self.detect_loss = torch.nn.BCELoss()
        
        self.automatic_optimization = False
        
        self.val_index_loader = []
        self.val_score_loader = []
        self.eval_index_loader = []
        self.eval_score_loader = []
        self.eval_filename_loader = []
        self.default_monitor = "val_eer"

    def forward(self, batch, is_train=True):
        if is_train:
            x, feat, target = batch
        else:
            if len(batch) == 4:
                x, feat, target, audio_path = batch
            else:
                x, feat, target = batch
                audio_path = None
        x_wav = get_input(x)
        with torch.no_grad():
            self.decouple_model.eval()
            G_x, commit_loss, last_layer, acoustic_tokens = self.decouple_model(x_wav, layers=[0,1,2,3,4,5,6,7])
        raw_logits, raw_feature = self.detect_model(acoustic_tokens)
        
        if is_train:
            onehot_target = torch.eye(2).to(self.device)[target, :]
            raw_logits = torch.softmax(raw_logits, dim=-1)
            raw_former_loss_ = self.detect_loss(raw_logits,onehot_target)
            return raw_former_loss_, raw_logits, target
        else:
            raw_logits = torch.softmax(raw_logits, dim=-1)[:, 0]
            raw_former_loss_ = 0
            return audio_path, raw_former_loss_, raw_logits, target
        
    def training_step(self, batch, batch_idx):
        raw_opt = self.optimizers()
        
        raw_former_loss_, raw_logits, target = self(batch, is_train=True)
        raw_opt.zero_grad()
        self.manual_backward(raw_former_loss_)
        raw_opt.step()
        
        self.log_dict(
            {   
                'train_loss': raw_former_loss_
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True)
    
    def validation_step(self, batch, batch_idx):
        _, raw_former_loss_, raw_logits, target = self(batch, is_train=False)
        
        self.val_index_loader.append(target)
        self.val_score_loader.append(raw_logits)
        
        self.log_dict(
            {
                'val_loss': raw_former_loss_,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True)
        
    def on_validation_epoch_end(self):
        all_index = self.all_gather(torch.cat(self.val_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.val_score_loader, dim=0)).view(-1).cpu().numpy()
        val_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_val_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])[0]
        val_eer = min(val_eer, other_val_eer)
        self.log_dict(
            {
                "val_eer": val_eer,
            },
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        
        self.val_index_loader.clear()  # free memory
        self.val_score_loader.clear()  # free memory
        
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]['lr'],
            },
            sync_dist=True,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )
        
        adjust_learning_rate(self.optimizers(), self.current_epoch, self.lr_raw_former, self.trainer.max_epochs*0.1, self.trainer.max_epochs)
        
    def test_step(self, batch, batch_idx):
          
        audio_path, raw_former_loss_, raw_logits, target = self(batch, is_train=False)
        
        self.eval_index_loader.append(target)
        self.eval_score_loader.append(raw_logits)
        self.eval_filename_loader.append(audio_path)
        self.log_dict(
            {
                'val_loss_rawformer': raw_former_loss_,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True)
        
    def on_test_epoch_end(self):
        
        string_list = [list(item) for item in self.eval_filename_loader]
    
        all_filename = np.array(string_list)
        all_filename = all_filename.reshape(-1, 1)
        
       
        all_index = self.all_gather(torch.cat(self.eval_index_loader, dim=0)).view(-1).cpu().numpy()
        all_score = self.all_gather(torch.cat(self.eval_score_loader, dim=0)).view(-1).cpu().numpy()
        
        # gpu_id = torch.cuda.current_device()
        
        data_to_write = zip(all_filename, all_score,all_index)
        csv_filename = self.save_score_path + '/score.csv'
        eval_eer = compute_eer(all_score[all_index == 0], all_score[all_index == 1])[0]
        other_eval_eer = compute_eer(-all_score[all_index == 0], -all_score[all_index == 1])[0]   
        eval_eer = min(eval_eer, other_eval_eer)
        
        self.log_dict(
            {
                "test_eer": eval_eer,
            },
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        
        self.eval_index_loader.clear()  # free memory
        self.eval_score_loader.clear()  # free memory
        self.eval_filename_loader.clear()  # free memory
        
    def configure_optimizers(self):
        optimizer_rawformer = torch.optim.AdamW(self.detect_model.parameters(), lr=self.lr_raw_former, weight_decay=1e-4)
        
        return [optimizer_rawformer]
        
def adjust_learning_rate(optimizer, epoch, lr, warmup, epochs=100):
    lr = lr
    if epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi *
                     (epoch - warmup) / (epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr