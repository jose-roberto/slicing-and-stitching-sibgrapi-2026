import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.utils import initialize_unetr_weights

class PatchEmbedding(nn.Module):
    
    def __init__(self, img_size, patch_size, in_channels, embedding_dim):
        
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        
        self.n_patches = (
            img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]
        )

        self.proj_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        
        x = self.proj_conv(x)
       
        x = x.flatten(2)
        
        x = x.transpose(1, 2)
        
        return x # [n_samples, n_patches, embedding_dim]

class Attention(nn.Module):
    
    def __init__(self, dim, n_heads, qkv_bias, attn_dropout, proj_dropout):

        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        
    def forward(self, x):
        
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError 
        
        qkv = self.qkv(x) # [n_samples, n_patches, 3 * dim]

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # [n_samples, n_patches, 3, n_heads, head_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, n_samples, n_heads, n_patches, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2] # [n_samples, n_heads, n_patches, head_dim]

        dropout_p = self.attn_dropout.p if self.training else 0.0
        weighted_avg = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        weighted_avg = weighted_avg.transpose(1, 2) # [n_samples, n_patches, n_heads, head_dim]
        weighted_avg = weighted_avg.flatten(2) # [n_samples, n_patches, dim]

        x = self.proj(weighted_avg) # [n_samples, n_patches, dim]
        x = self.proj_dropout(x) # [n_samples, n_patches, dim]

        return x
    
class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        
        super().__init__()

        self.fully_connected_1 = nn.Linear(in_features, hidden_features)
        self.gelu_act = nn.GELU()
        self.fully_connected_2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.fully_connected_1(x)

        x = self.gelu_act(x)

        x = self.fully_connected_2(x)

        x = self.dropout(x)

        return x # [n_samples, n_patches, dim]

class TransformerBlock(nn.Module):
    
    def __init__(self, dim, n_heads, qkv_bias, attn_dropout, proj_dropout, mlp_ratio):
        
        super().__init__()

        self.normalization_1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout
        )

        self.normalization_2 = nn.LayerNorm(dim, eps=1e-6)

        self.hidden_features = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=self.hidden_features,
            out_features=dim
        )
        
    def forward(self, x):

        x = x + self.attn(self.normalization_1(x))
        
        x = x + self.mlp(self.normalization_2(x))
        
        return x

class ViT(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size,
        patch_size=(16, 16, 16),
        embedding_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        p_dropout=0.,
        attn_dropout=0.,
        skip_layer_indices=[2, 5, 8]
    ):
        
        super().__init__()

        self.skip_layer_indices = skip_layer_indices

        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.n_patches, embedding_dim)
        )

        torch.nn.init.trunc_normal_(self.position_embedding, std=0.02) 

        self.position_dropout = nn.Dropout(p=p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embedding_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_dropout,
                    proj_dropout=p_dropout
                )
                for _ in range(depth)
            ]
        )

        self.normalization = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, x):

        n_samples = x.shape[0]

        x = self.patch_embedding(x)

        x = x + self.position_embedding

        x = self.position_dropout(x)

        skip_connection_outputs = []

        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x)

            if i in self.skip_layer_indices:
                skip_connection_outputs.append(x)
                
        x = self.normalization(x)
        
        return x, skip_connection_outputs   

class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_layers):
        
        super().__init__()
        
        layers = []
        
        current_channels = in_channels

        for i in range(n_layers):
            layers.append(nn.Conv3d(current_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding, bias=False))
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
            layers.append(nn.LeakyReLU())
            
            current_channels = out_channels

        self.convolution_block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.convolution_block(x)

class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            bias=False
        )
        
    def forward(self, x):
        
        return self.upsample(x)

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_layers):
        
        super().__init__()

        self.leaky_relu_act = nn.LeakyReLU()

        self.convolution_block = ConvolutionBlock(in_channels, out_channels, kernel_size, stride, padding, n_layers)

        if in_channels == out_channels:
            self.shortcut_conv = nn.Identity()
        else:
            self.shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        
        out = self.convolution_block(x) + self.shortcut_conv(x)

        return self.leaky_relu_act(out)
        
class UNETR(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size,
        patch_size=(16, 16, 16),
        embedding_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        p_dropout=0.,
        attn_dropout=0.
    ):
        
        super().__init__()
        
        self.depth = depth
        
        self.feature_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])

        self.skip_layer_indices = [2, 5, 8]

        self.vit = ViT(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            skip_layer_indices=self.skip_layer_indices
        )
        
        decoder_channels = [256, 128, 64, 32]
        
        # Entrada comum aos skips (após reshape_seq_to_vol):
        # [B, 768, H/16, W/16, D/16]

        self.skip_z9_proc = nn.Sequential(
            UpsampleBlock(embedding_dim, decoder_channels[0], kernel_size=2, stride=2), # [B, 256, H/8, W/8, D/8]
            ConvolutionBlock(decoder_channels[0], decoder_channels[0], kernel_size=3, stride=1, padding=1, n_layers=1) # [B, 256, H/8, W/8, D/8]
        )

        self.skip_z6_proc = nn.Sequential(
            UpsampleBlock(embedding_dim, decoder_channels[1], kernel_size=2, stride=2), 
            ConvolutionBlock(decoder_channels[1], decoder_channels[1], kernel_size=3, stride=1, padding=1, n_layers=1),
            UpsampleBlock(decoder_channels[1], decoder_channels[1], kernel_size=2, stride=2), 
            ConvolutionBlock(decoder_channels[1], decoder_channels[1], kernel_size=3, stride=1, padding=1, n_layers=1) 
        )
        
        self.skip_z3_proc = nn.Sequential(
            UpsampleBlock(embedding_dim, decoder_channels[2], kernel_size=2, stride=2), 
            ConvolutionBlock(decoder_channels[2], decoder_channels[2], kernel_size=3, stride=1, padding=1, n_layers=1),
            UpsampleBlock(decoder_channels[2], decoder_channels[2], kernel_size=2, stride=2), 
            ConvolutionBlock(decoder_channels[2], decoder_channels[2], kernel_size=3, stride=1, padding=1, n_layers=1),
            UpsampleBlock(decoder_channels[2], decoder_channels[2], kernel_size=2, stride=2), 
            ConvolutionBlock(decoder_channels[2], decoder_channels[2], kernel_size=3, stride=1, padding=1, n_layers=1) 
        )
        
        self.bottleneck = UpsampleBlock(embedding_dim, decoder_channels[0], kernel_size=2, stride=2) # [B, 768, H/16] -> [B, 256, H/8]
        self.residual_block_1 = ResidualBlock(decoder_channels[0] * 2, decoder_channels[0], kernel_size=3, stride=1, padding=1, n_layers=2) # [B, 512, H/8] -> [B, 256, H/8]

        self.upsample_block_1 = UpsampleBlock(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2) # [B, 256, H/8] -> [B, 128, H/4]
        self.residual_block_2 = ResidualBlock(decoder_channels[1] * 2, decoder_channels[1], kernel_size=3, stride=1, padding=1, n_layers=2) # [B, 256, H/4] -> [B, 128, H/4]

        self.upsample_block_2 = UpsampleBlock(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2) # [B, 128, H/4] -> [B, 64, H/2]
        self.residual_block_3 = ResidualBlock(decoder_channels[2] * 2, decoder_channels[2], kernel_size=3, stride=1, padding=1, n_layers=2) # [B, 128, H/2] -> [B, 64, H/2]
        
        self.upsample_block_3 = UpsampleBlock(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2) # [B, 64, H/2] -> [B, 32, H]
        
        self.input_residual_block = ResidualBlock(in_channels, decoder_channels[3], kernel_size=3, stride=1, padding=1, n_layers=2) # [B, in_channels, H] -> [B, 32, H]
        
        self.output_residual_block = ResidualBlock(decoder_channels[3] * 2, decoder_channels[3], kernel_size=3, stride=1, padding=1, n_layers=2) # [B, 64, H] -> [B, 32, H]
        
        self.output_conv = nn.Conv3d(decoder_channels[3], out_channels, kernel_size=1, stride=1) # [B, 32, H] -> [B, out_channels, H]

        initialize_unetr_weights(self)

    def reshape_seq_to_vol(self, x):
        
        b, n, d = x.shape

        x = x.transpose(1, 2)

        x = x.view(b, d, self.feature_size[0], self.feature_size[1], self.feature_size[2])
        
        return x # [B, N, D] -> [B, D, H, W, D]

    def forward(self, x):
        
        input_skip = x 
    
        z12, skip_outputs = self.vit(x)
        
        z3, z6, z9 = skip_outputs[0], skip_outputs[1], skip_outputs[2]

        z12_vol = self.reshape_seq_to_vol(z12)
        z9_vol  = self.reshape_seq_to_vol(z9) 
        z6_vol  = self.reshape_seq_to_vol(z6) 
        z3_vol  = self.reshape_seq_to_vol(z3) 
        
        up_z12 = self.bottleneck(z12_vol) # [B, 256, H/8, W/8, D/8]
                
        skip_z9 = self.skip_z9_proc(z9_vol) # [B, 256, H/8, W/8, D/8]

        concat_1 = torch.cat([up_z12, skip_z9], dim=1) # [B, 512, H/8, W/8, D/8]
        res_1 = self.residual_block_1(concat_1) # [B, 256, H/8, W/8, D/8]
        
        up_res_1 = self.upsample_block_1(res_1) # [B, 128, H/4, W/4, D/4]

        skip_z6 = self.skip_z6_proc(z6_vol) # [B, 128, H/4, W/4, D/4]

        concat_2 = torch.cat([up_res_1, skip_z6], dim=1) # [B, 256, H/4, W/4, D/4]
        res_2 = self.residual_block_2(concat_2) # [B, 128, H/4, W/4, D/4]

        up_res_2 = self.upsample_block_2(res_2) # [B, 64, H/2, W/2, D/2]

        skip_z3 = self.skip_z3_proc(z3_vol) # [B, 64, H/2, W/2, D/2]

        concat_3 = torch.cat([up_res_2, skip_z3], dim=1) # [B, 128, H/2, W/2, D/2]
        res_3 = self.residual_block_3(concat_3) # [B, 64, H/2, W/2, D/2]

        up_res_3 = self.upsample_block_3(res_3) # [B, 32, H, W, D]
        
        skip_input = self.input_residual_block(input_skip) # [B, 32, H, W, D]
        
        concat_4 = torch.cat([up_res_3, skip_input], dim=1) # [B, 64, H, W, D]
        res_4 = self.output_residual_block(concat_4) # [B, 64, H, W, D]
        
        output = self.output_conv(res_4) # [B, out_channels, H, W, D]
        
        return output
        