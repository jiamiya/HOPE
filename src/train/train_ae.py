import sys
sys.path.append("..")
sys.path.append(".")
import time
# import os
# os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# os.environ["TORCH_NNPACK"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model.network import VAE_Conv, AE_Conv
from data_collection.dataset import ParkingImgDataset
from configs import *

EPOCH = 10
data_path = "./parking/use_img_no_tail"

def Loss(reconstruct_batch, origin_batch, mu, log_var):
    loss_KLD = torch.mean(0.5 * torch.sum(mu.pow(2)- log_var + log_var.exp()-1))#torch.sum(-log_var + 0.5*(log_var.exp().pow(2) + mu.pow(2)))#
    # loss_BCE = F.binary_cross_entropy(reconstruct_batch, origin_batch, reduction='sum')
    loss_BCE = F.mse_loss(reconstruct_batch, origin_batch)
    loss = loss_KLD + loss_BCE
    return loss, loss_KLD, loss_BCE

def update_vae(data, model, optimizer):
    # TODO: data type modified
    random.shuffle(data)
    train_data = data[:-100]
    eval_data = data[-100:]
    for e in range(EPOCH):
        loss_list = []
        loss_BCE_list = []
        random.shuffle(train_data)
        for img in train_data:
            # img = np.ones((2,64,64))
            img_input = torch.FloatTensor(img).to(device).unsqueeze(0)
            # print(img_input)
            # print("*")
            img_output, mean, std = model(img_input)
            # import matplotlib.pyplot as plt
            # plt.imshow(img_output[0][0].detach().cpu().numpy())
            # plt.show()
            # print(img_output)
            # print(mean)
            # print(std)
            # p
            loss, loss_KLD, loss_BCE = Loss(img_output, img_input, mean, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(abs(loss.cpu().item()))
            loss_BCE_list.append(loss_BCE.item())
        print(e,np.mean(loss_list), np.mean(loss_BCE_list), np.mean(loss_list)-np.mean(loss_BCE_list))
        # print(mean)
        # print(std)
        print(torch.mean(mean).item(), torch.std(mean).item(), torch.mean(std).item(), torch.std(std).item(), '\n')
    model.eval()
    total_loss = 0
    for i in range(len(eval_data)):
        img_input = torch.FloatTensor(eval_data[i]).to(device).unsqueeze(0)
        img_output, mean, std = model(img_input)
        loss, loss_KLD, loss_BCE = Loss(img_output, img_input, mean, std)
        total_loss += loss.cpu().item()/len(eval_data)
        if i%10 == 0:
            print(mean,std)
        #     print('loss:', loss.item())
            # plt.figure()
            # plt.subplot(2,1,1)
            # plt.imshow(eval_data[i].transpose(1,2,0))
            # plt.subplot(2,1,2)
            # plt.imshow(img_output[0].detach().cpu().numpy().transpose(1,2,0))
            # plt.show()
    print(total_loss)

def normalize_img(imgs):
    imgs_mean = np.mean(imgs, axis=0)
    imgs_std = np.std(imgs, axis=0)
    normed_imgs = [(img -imgs_mean)/(imgs_std+1e-6) for img in imgs]
    return normed_imgs, imgs_mean, imgs_std

def update_ae(data, model, optimizer):
    print(">> initializing train dataset index")
    data_idx = list(np.arange(len(data)))
    random.shuffle(data_idx)
    train_num = int(len(data)*0.9)
    train_data_idx = data_idx[:train_num]
    print(">> Start Trainging !!!")
    for e in range(EPOCH):
        loss_list = []
        random.shuffle(train_data_idx)
        for img_idx in tqdm(train_data_idx):
            # img = np.ones((2,64,64))
            img = data[img_idx]
            img_input = torch.FloatTensor(img).to(device).unsqueeze(0)
            # print(img_input)
            # print("*")
            img_output = model(img_input)
            # import matplotlib.pyplot as plt
            # plt.imshow(img_output[0][0].detach().cpu().numpy())
            # plt.show()
            # print(img_output)
            # print(mean)
            # print(std)
            # p
            # loss = F.binary_cross_entropy(img_output, img_input, reduction='mean')
            loss = F.mse_loss(img_output, img_input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(abs(loss.cpu().item()))
            # p
        print("epoch %s : "%e, np.mean(loss_list))

    print(">> EVAL ")
    print(">> initializing eval dataset index")
    eval_data_idx = data_idx[train_num:]
    eval_data = [data[i] for i in eval_data_idx]
    model.eval()
    total_loss = 0
    for i in range(len(eval_data)):
        img_input = torch.FloatTensor(eval_data[i]).to(device).unsqueeze(0)
        img_output = model(img_input)
        loss = F.mse_loss(img_output, img_input)
        total_loss += loss.cpu().item()/len(eval_data)
        if i<5:
            print(model.embed(img_input))
        #     print('loss:', loss.item())
            # plt.figure()
            # plt.subplot(2,1,1)
            # plt.imshow(eval_data[i].transpose(1,2,0))
            # plt.subplot(2,1,2)
            # plt.imshow(img_output[0].detach().cpu().numpy().transpose(1,2,0))
            # plt.show()
    print(total_loss)



if __name__=="__main__":
    save = True
    verbose = False

    img_dataset = ParkingImgDataset(data_path, dataset_size=250000, load_to_memory=True)
    print("dataset size:", len(img_dataset))

    # the path to log and save model
    relative_path = '.'#os.path.dirname(os.getcwd())
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/ae_nt/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)


    seed = SEED
    np.random.seed(seed)
    torch.manual_seed(seed)

    # feature_size1 = env.lidar.lidar_num
    # feature_size2 = env.tgt_repr_size
    EMBD_SIZE = 128
    HIDD_SIZE = 256
    use_tanh = True
    vae = AE_Conv(img_dataset.get_img_shape(), 3, 128, C_CONV, SIZE_FC, use_tanh=use_tanh).to(device)
    # print(vae.parameters)
    optimizer = optim.Adam(vae.parameters(),lr=1e-4)

    if isinstance(vae, VAE_Conv):
        update_vae(img_dataset, vae, optimizer)
        vae.save(save_path+'vae_%s.pt'%use_tanh)
    else:
        update_ae(img_dataset, vae, optimizer)
        vae.save(save_path+'ae_nt_%s.pt'%use_tanh)
            