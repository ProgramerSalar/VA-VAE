import torch 
import pytorch_lightning as pl 
import torch.nn.functional as F
import torch.torch_version 
from vae.unet import Encoder, Decoder
import importlib
from vae.distribution import DiagonalGaussianDistribution
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vae.utils.utils import instantiate_from_config


class AutoencoderKL(pl.LightningModule):

    """ 
    A Variational Autoencoder (VAE) with KL divergance regularization and optional adversarial training.

    Args:
        ddconfig (dict): Configuration dictionary for encoder/decoder 
        emb_dim (int): Dimension of the latent embedding space 
        loss_config (dict): Configuration for the loss function 
        ckpt_path (str, optional): Path to checkpoint for loading pretrained weights 
        ignore_keys (list, optional): Keys to ignore when loading from checkpoint
        image_key (str): Key for accessing images in the batch dictionary 
        colorize_nlabels (int, optional): Number of labels for segmentation colorization
        monitor (str, optional): Metric to monitor for checkpointing
    """



    def __init__(self,
                 ddconfig,
                 emb_dim,
                 loss_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None):
        

        super().__init__()
        self.image_key = image_key
        self.learning_rate = 4.5e-6  # Default learning rate 
        self.automatic_optimization = False 

        # encoder network
        self.encoder = Encoder(**ddconfig)
        # decoder network
        self.decoder = Decoder(**ddconfig)
        # Loss function (typically LPIPSWithDiscriminator)
        self.loss = instantiate_from_config(loss_config)

        # Verify double_z is enable in config 
        assert ddconfig["double_z"], "Config must have double_z=True for variance output"

        # Projects encoder output to latent space 
        self.quant_conv = torch.nn.Conv2d(in_channels=2*ddconfig["z_channels"],  # Far mean and variance
                                          out_channels=2*emb_dim,               # Project to embedding dim
                                          kernel_size=1)
        
        # Projects latent back to decoder input 
        self.post_quant_conv = torch.nn.Conv2d(in_channels=emb_dim,
                                               out_channels=ddconfig["z_channels"],
                                               kernel_size=1)
        

        


    

    def encode(self, x):
        """Encode input into latent distribution parameters"""
        h = self.encoder(x)  # get encoder features
        moments = self.quant_conv(h)  # Projects to latent space parameters
        posterior = DiagonalGaussianDistribution(moments)   # Create distribution
        return posterior
    

    def decode(self, z):
        """Decode latent samples into reconstructions"""
        z = self.post_quant_conv(z)   # Project latent to decoder input dims
        dec = self.decoder(z)       # decode to image space 
        return dec 


    def forward(self, input, sample_posterior=True):
        """Full forward pass with optional sampling"""
        posterior = self.encode(input)   # Get latent distribution

        # sample from posterior or take mode (deterministic)
        z = posterior.sample() if sample_posterior else posterior.mode()

        dec = self.decode(z)
        return dec, posterior
    

    def training_step(self, batch, batch_idx):
        print("Batch shape:", batch)
        x = batch
        # print("x: ", x)
        # Forward pass 
        reconstructions, posterior = self(x)

        # print("Expected Loss function arguments: ", self.loss.forward.__code__.co_varnames)
        

        # calculate loss 
        loss_dict = self.loss(
            inputs=x,
            recontructions=reconstructions,
            posteriors=posterior,
            optimizer_idx=1,
            global_step=1,
            split="train"
        )

        # Log losses
        for k, v in loss_dict[1].items():
            self.log(f"train/{k}", v, prog_bar=True)
            
        return loss_dict[0]


    def configure_optimizers(self):
        # Create optimizers
        lr = self.hparams.get('lr', 4.5e-6)
        opt_ae = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.quant_conv.parameters()) + 
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        opt_disc = optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        
        # Create scheduler
        scheduler_ae = {
            'scheduler': ReduceLROnPlateau(opt_ae, mode='min', factor=0.5, patience=5),
            'monitor': 'val/rec_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [opt_ae, opt_disc], [scheduler_ae]






    

    
    




        