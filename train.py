from vae.utils.utils import load_config
from .dataset import VQVAEDataset
from torch.utils.data import DataLoader
from .autoencoder import VQModelInterface
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
import torch 




def main():

    config = "--config_path--"
    config = load_config(config_path=config)
    print("COnfig: ", config)
    
    train_dataset = VQVAEDataset(root_dir=config['data']['dataset']['image_folder'],
                                split="train",
                                image_size=256)
    

    val_dataset = VQVAEDataset(root_dir=config['data']['dataset']['image_folder'],
                                split="val",
                                image_size=256)
    
    train_datloader = DataLoader(dataset=train_dataset,
                               batch_size=config['data']['dataloader']['batch_size'],
                               shuffle=config['data']['dataloader']['shuffle'])
    
    val_datloader = DataLoader(dataset=val_dataset,
                               batch_size=config['data']['dataloader']['batch_size'],
                               shuffle=config['data']['dataloader']['shuffle'])
    
    

    
    model = VQModelInterface(
        embed_dim=config['model']['params']['embed_dim'],
        ddconfig=config['model']['params']['ddconfig'],
        n_embed=config['model']['params']['n_embed'],
        lossconfig=config['model']['params']['lossconfig'],
        monitor='val/total_loss',
        use_ema=True
    )

    callbacks = [
        ModelCheckpoint(
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            dirpath="./checkpoints",
            filename='vae-{epoch:02d}-{val_loss:.2f}',
            save_last=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]


    # trainer configuration 
    trainer = Trainer(
        max_epochs=1,
        callbacks=callbacks,
        devices=1 if torch.cuda.is_available() else None,  # More flexible device handling
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision=32,  # Can use '16-mixed' for FP16
        accumulate_grad_batches=1
    )

    # Training 
    trainer.fit(model, train_datloader, val_datloader)
    print("Training completed!")

    
if __name__ == "__main__":
    main()