from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import cat

from model.attention import AttentionNetwork

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Network(nn.Module):
    def __init__(self, layers: list, orthogonal_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(layers))
        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

    def forward(self, x):
        out = self.net(x)
        return out

class MultiObsEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        embed_size = configs['embed_size']
        hidden_size = configs['hidden_size']
        activate_func = [nn.LeakyReLU(), nn.Tanh()][configs['use_tanh_activate']]
        self.use_img = False if configs['img_shape'] is None else True
        self.use_action_mask = False if configs['action_mask_shape'] is None else True
        self.use_attention = False if configs['attention_configs'] is None else True
        self.input_action = 'input_action_dim' in configs and configs['input_action_dim'] > 0

        if not self.use_attention:
            if configs['n_hidden_layers'] == 1:
                layers = [nn.Linear(configs['n_modal']*embed_size, configs['output_size'])]
            else:
                layers = [nn.Linear(configs['n_modal']*embed_size, hidden_size)]
                for _ in range(configs['n_hidden_layers']-2):
                    layers.append(activate_func)
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Linear(hidden_size, configs['output_size']))
            self.net = nn.Sequential(*layers)
        else:
            attention_configs = configs['attention_configs']
            self.net = AttentionNetwork(
                embed_size,
                attention_configs['depth'],
                attention_configs['heads'],
                attention_configs['dim_head'],
                attention_configs['mlp_dim'],
                configs['n_modal'],
                attention_configs['hidden_dim'],
                configs['output_size'],
            )
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None

        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)

        if configs['target_shape'] is not None:
            layers = [nn.Linear(configs['target_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_tgt = nn.Sequential(*layers)
            
        if configs['action_mask_shape'] is not None:
            layers = [nn.Linear(configs['action_mask_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_am = nn.Sequential(*layers)

        if configs['img_shape'] is not None:
            self.embed_img = ImgEncoder(configs['img_shape'], configs['k_img_conv'],\
                                    embed_size, configs['img_conv_layers'], configs['img_linear_layers'])
            self.re_embed_img = nn.Sequential(activate_func, nn.Linear(embed_size, embed_size)) # the latten vector may not be scaled

        if self.input_action:
            layers = [nn.Linear(configs['input_action_dim'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_action = nn.Sequential(*layers)

        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight") and len(layer.shape)>1:
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_lidar.state_dict().items():
            # The output layer is specially dealt
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_tgt.state_dict().items():
            # The output layer is specially dealt
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        if self.use_action_mask:
            for layer_name, layer in self.embed_am.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
        
        if self.use_img:
            for layer_name, layer in self.re_embed_img.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
        
        if self.input_action:
            for layer_name, layer in self.embed_action.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)

    def load_img_encoder(self, path, device, require_grad = False):
        ae = torch.load(path, map_location=device)
        self.embed_img = ae.encoder
        for param in self.embed_img.parameters():
            param.requires_grad = require_grad

    def forward(self, x:dict):
        '''
            x: dictionary of different input modal. Includes:

            `img` : image with shape (n, c, w, h)
            `target` : tensor in shape (n, t)
            `lidar` : tensor in shape (n, l)

        '''
        feature_lidar = self.embed_lidar(x['lidar'])
        feature_target = self.embed_tgt(x['target'])
        features = [feature_lidar, feature_target]
        if self.use_action_mask:
            feature_am = self.embed_am(x['action_mask'])
            features.append(feature_am)

        if self.use_img:
            feature_img, _ = self.embed_img(x['img'])
            feature_img = self.re_embed_img(feature_img)
            features.append(feature_img)

        if self.input_action:
            feature_action = self.embed_action(x['action'])
            features.append(feature_action)

        if self.use_attention:
            embed = torch.stack(features, dim=1)
        else:
            embed = cat(features, dim=1)
        out = self.net(embed)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, Cin, Cout, K, Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        # self.downSample = nn.AvgPool2d(2)
        self.shortcut = nn.Sequential(
                     nn.Conv2d(Cin, Cout, kernel_size=1),
                     nn.AvgPool2d(2)
                )
        self.res = Res
        self.cin = Cin
        self.cout = Cout
    
    def forward(self,x):
        x1 = self.layer(x)
        if self.res:
            x_res = self.shortcut(x)
            x1 = x1 + x_res
        return x1
    
class DeConvBlock(nn.Module):
    def __init__(self, Cin, Cout, K, upsample, padding=None, Batch_norm=False, Res = True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )

        self.res = Res
        self.cin = Cin
        self.cout = Cout
        self.short_cut = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=1),
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=1),
            )

    def forward(self,x):
        x1 = self.layer(x)
        _, _, W, H = x.shape
        if W != H:
            raise NotImplementedError
        if self.res:
            x_res = self.short_cut(x)
            x1 = x1 + x_res
        return x1

class ImgEncoder(nn.Module):
    def __init__(self, input_shape, K, embed_size, c_conv_list, size_fc_list,\
                Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        Cin, w, h = input_shape
        layers = [ConvBlock(Cin, c_conv_list[0], K, Pooling, padding, Batch_norm, Res, use_tanh)]
        for i in range(len(c_conv_list)-1):
            layers.append(ConvBlock(c_conv_list[i], c_conv_list[i+1], K, Pooling, padding, Batch_norm, Res, use_tanh))
        layers.append(nn.Flatten())
        linear_input_shape = int(w*h*c_conv_list[-1]/(4**(len(c_conv_list))))
        layers.extend([nn.Linear(linear_input_shape, size_fc_list[0]), activate_func])
        for i in range(len(size_fc_list)-1):
            layers.append(nn.Linear(size_fc_list[i], size_fc_list[i+1]))
            layers.append(activate_func)
        self.net = nn.Sequential(*layers)
        self.output_mean = nn.Linear(size_fc_list[-1],embed_size)
        self.output_std = nn.Linear(size_fc_list[-1],embed_size)
    
    def forward(self, x):
        x = self.net(x)
        return self.output_mean(x), self.output_std(x)
    
class ImgDecoder(nn.Module):
    def __init__(self, output_shape, K, embed_size, c_conv_list, size_fc_list,\
                 padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        out_channel, w, h = output_shape
        self.output_shape = output_shape
        self.c_conv_list = c_conv_list
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]

        # The fully connected layers
        fc_list = []
        fc_list.extend([nn.Linear(embed_size, size_fc_list[-1]), activate_func])
        for i in range(len(size_fc_list)-1):
            fc_list.append(nn.Linear(size_fc_list[-(i+1)], size_fc_list[-(i+2)]))
            fc_list.append(activate_func)
        fc_output_size = int( h/(2**len(c_conv_list)) * w/(2**len(c_conv_list)) * c_conv_list[-1] )
        fc_list.extend([nn.Linear(size_fc_list[0], fc_output_size), activate_func])
        self.fc_net = nn.Sequential(*fc_list)

        # The DeConvolution layers
        conv_list = []
        upsample_size = int(h/(2**len(c_conv_list))*2)
        for i in range(len(c_conv_list)-1):
            conv_list.append(DeConvBlock(c_conv_list[-i-1],c_conv_list[-i-2],K,upsample_size, padding, Batch_norm, Res, use_tanh))
            upsample_size *= 2
        conv_list.append(DeConvBlock(c_conv_list[0],out_channel,K,upsample_size, padding, Batch_norm, Res, use_tanh))
        self.conv_net = nn.Sequential(*conv_list)

        self.output = nn.Sigmoid()
    
    def forward(self, z):
        x = self.fc_net(z)
        B = x.shape[0]
        # assume the image: w==h
        w = int(self.output_shape[-2]/(2**len(self.c_conv_list)))
        x = x.reshape((B, self.c_conv_list[-1], w, w))
        x = self.conv_net(x)
        x = self.output(x)
        return x
    
class VAE_Conv(nn.Module):
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        super(VAE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        sampled_z = self.sampling(mean,std)
        return self.decoder(sampled_z), mean, std
    
    def eval_forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        # sampled_z = self.sampling(mean,std)
        return self.decoder(mean), mean, std

    def embed(self,img):
        '''
        Input:
            img: torch.Tensor of shape (B, C, W, H)
        Return:
            embed_img: torch.Tensor of Shape (B, HIDDEN_SIZE)
        '''
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img,_ = self.encoder(img)
        return embed_img
    
    def save(self, path):
        torch.save(self, path)
        print('save model in: %s'%path)

class AE_Conv(nn.Module):
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        super(AE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,_ = self.encoder(x)
        return self.decoder(mean)

    def embed(self,img):
        '''
        Input:
            img: torch.Tensor of shape (B, C, W, H)
        Return:
            embed_img: torch.Tensor of Shape (B, HIDDEN_SIZE)
        '''
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img, embed_std = self.encoder(img)
        return embed_img, embed_std

    def save(self, path):
        torch.save(self, path)
        print('save model in: %s'%path)