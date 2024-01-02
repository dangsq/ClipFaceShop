import argparse
import os

import numpy
import random
import torch
import torchvision
from torch import optim
from tqdm import tqdm
from criteria.clip_loss import CLIPLoss
from models.stylegan2.model import Generator
import math
import torchvision.transforms as transforms
from PIL import Image
from argparse import Namespace
from models.psp import pSp
import dlib
from utils.alignment import align_face
from experiments.Losses import IDLoss
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch_utils.model_utils import unet
import warnings
import pickle


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

warnings.filterwarnings("ignore")

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform = transforms.Compose([
    Resize(224, interpolation=Image.Resampling.BICUBIC),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

def get_latent(g_ema):
    mean_latent = g_ema.module.mean_latent(4096).cuda()
    latent_code_init_not_trunc = torch.randn(1, 512).cuda()
    with torch.no_grad():
        _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent)

    direction = latent_code_init.detach().clone().cuda()
    direction.requires_grad = True
    return direction


def load_model(args):
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = torch.nn.DataParallel(g_ema)
    g_ema = g_ema.cuda()
    return g_ema


def get_lr(t, initial_lr, rampdown=0.75, rampup=0.005):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents
def transtopeeps(a):
    s=numpy.array(a.convert('RGBA'))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if not numpy.sum(s[i][j]-[80,106,244,255]):
                s[i][j]=numpy.array([255,255,255,255])
    a = Image.fromarray(s).convert('L')
    threshold = 190

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    return a.point(table,'1')
def main(args):
    
    g_ema = load_model(args)
    id_loss = IDLoss().cuda().eval()
    print('Loading face parsing net...')
    G_mask=unet().cuda()
    G_mask.load_state_dict(torch.load('./pretrained_models/mask_net.pth'))
    
    if args.dir_name is None:
        name_style = os.path.splitext(os.path.basename(args.target_path))[0]
        args.dir_name = name_style
    if args.output_folder is not None:
        args.dir_name = args.output_folder + args.dir_name
    dir_name = args.dir_name

    if not os.path.exists(dir_name):
        try:
            os.mkdir(dir_name)
        except:
            os.makedirs(dir_name)


    NUM_IMAGES = args.num_images
    
    directions = [get_latent(g_ema)]
        
    directions_cat = torch.cat(directions)
        
    
    mpp=numpy.load('./dirs/get_w/allc_vistmp.npy')
    mask=torch.zeros(18)
    mask_index=list(numpy.abs(mpp.mean(axis=1)).argsort()[::-1]) 
    mask[mask_index[0:args.mpp_d]]+=1   
    mask=mask.reshape(1,18,1).cuda()
    mask.requires_grad=False

    maskA=mask
    maskAc=1-maskA
    maskA.requires_grad=False

    maskAc.requires_grad=False
    
    with torch.no_grad():
        
        latents_s = [None] * 4

        latents_0=torch.load(args.train_latents_path)

        try:
            data = torch.load(args.test_latents_path) 
        except:
            data = torch.tensor(numpy.load(args.test_latents_path)).cuda() 
        latents_0_save=torch.zeros(latents_0.shape)

        for n in range(4):
            with torch.no_grad():
                latents_s[n] = data[n].unsqueeze(0).cuda()
                latents_s[n].requires_grad = False
        latents = torch.cat(latents_s)
        with torch.no_grad():
            img_gen, _ = g_ema([latents], input_is_latent=True, randomize_noise=False)
            for i in range(latents.shape[0]):
                torchvision.utils.save_image(img_gen[i], f"{dir_name}/img_gen_{i}.png", normalize=True, range=(-1, 1))
    clip_loss = CLIPLoss(args.stylegan_size)
    clip_loss = torch.nn.DataParallel(clip_loss)

    At=torch.randn(directions[0].shape).to('cuda')

    At.requires_grad_()
    optimizer = optim.Adam(directions+[At], lr=args.lr, weight_decay=0)

    with torch.no_grad():
        
        targets_clip = None
        if args.target_path is not None:
            if args.toL==1:
                img_target = Image.open(args.target_path) 
            else:
                img_target = Image.open(args.target_path)
                
            img_target = transform(img_target).unsqueeze(0).cuda()

            torchvision.utils.save_image(img_target, f"{dir_name}/target.png",
                                         normalize=True, range=(-1, 1))
            target_clip = clip_loss.module.model.encode_image(img_target)
            
            target_clip = target_clip / target_clip.norm(dim=-1)
            base_clip=torch.load('./dirs/s/u_fc.pt').cuda().to(target_clip.dtype)
                 
            base_clip = base_clip / base_clip.norm(dim=-1)
            base_clip.requires_grad = False
            gap_clip = target_clip-base_clip

            if torch.abs(torch.sum(gap_clip))<1e-5:
                numpy.save('{0}/direction{1}{2}.npy'.format(dir_name, 0,args.str),
                               numpy.zeros((18,512)))
                print('gap is zero so this stage is skipped')
                return 0
            gap_clip = gap_clip / gap_clip.norm(dim=-1)
            
            target_clip.requires_grad = False
            base_clip.requires_grad = False
            gap_clip.requires_grad = False

        else:
            # target is latent dir
            with torch.no_grad():
                img_target, _ = g_ema([directions_cat], input_is_latent=True, randomize_noise=False)
                targets_clip = clip_loss.module.encode(img_target)
                targets_clip.requires_grad = False

    for dir_idx, direction in enumerate(directions):
        
        opt_loss = torch.Tensor([float("Inf")]).cuda()
        pbar = tqdm(range(args.step))
        losslist={}
        losslist['loss']=torch.zeros(args.step)
        losslist['tlloss']=torch.zeros(args.step)
        losslist['closs']=torch.zeros(args.step)
        losslist['gaploss']=torch.zeros(args.step)
        losslist['simloss']=torch.zeros(args.step)
        losslist['idloss']=torch.zeros(args.step)
        losslist['bkloss']=torch.zeros(args.step)
        best_direction=direction.clone().detach()
        best_At=At.clone().detach()
        best_latents=latents.clone().detach()
        
        
        for i in pbar:
            latents = latents_0[i*NUM_IMAGES:(i+1)*NUM_IMAGES]
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr


            loss = torch.zeros(1).cuda()
            target_semantic = torch.zeros(1).cuda()
            similarities_loss = torch.zeros(1).cuda()
            
            with torch.no_grad():
                img_gen0, _ = g_ema([latents], input_is_latent=True, randomize_noise=False)
                
                bk_mask=G_mask(img_gen0)
                

            direction_all = [direction for i in range(args.num_images)]
            
            direction_all = torch.stack(direction_all).squeeze(1).cuda()
            
            img_gen_amp0, _ = g_ema([((maskA)*At+maskAc*(torch.ones(1,18,512).cuda()))*latents+mask*direction_all], input_is_latent=True, randomize_noise=False)
                
            bk_mask_amp=G_mask(img_gen_amp0)
                
            bk_mask_img1=img_gen0*(torch.sigmoid(bk_mask[0,0])+torch.sigmoid(bk_mask[0,18]))

            bk_mask_amp_img1=img_gen_amp0*(torch.sigmoid(bk_mask_amp[0,0])+torch.sigmoid(bk_mask_amp[0,18]))
            
            
            img_gen=img_gen0
            img_gen_amp=img_gen_amp0
            
            image_gen_clip = clip_loss.module.encode(img_gen)
            image_gen_clip_norm = image_gen_clip / image_gen_clip.norm(dim=-1, keepdim=True)
            
            
            image_gen_amp_clip = clip_loss.module.encode(img_gen_amp)

            image_gen_amp_clip_norm = image_gen_amp_clip / image_gen_amp_clip.norm(dim=-1, keepdim=True)
            
            gap_image_gen_clip = image_gen_amp_clip_norm-image_gen_clip_norm
            
            gap_image_gen_clip_norm = gap_image_gen_clip / (gap_image_gen_clip.norm(dim=-1, keepdim=True))


            gap_s=torch.zeros(1).cuda()

            gap_gap = gap_image_gen_clip_norm @ (gap_clip).T
            
            gap_s += args.lambda_gaps*(1- gap_gap.mean())
            similarity_gap = image_gen_amp_clip_norm @ (target_clip).T
            target_l=args.lambda_transfer * (1 - similarity_gap.mean())
            target_semantic += target_l
            loss+=gap_s
            
            bkloss=torch.zeros(1).cuda()
            if args.lambda_background>0:

                bkloss+=args.lambda_background*torch.sum(torch.abs(bk_mask_amp_img1-bk_mask_img1))/NUM_IMAGES
                
            loss+=bkloss

            sourceLoss=torch.zeros(1).cuda()

            if args.lambda_sourcesim!=0.0:
                sourceLoss+=args.lambda_sourcesim*(1-torch.sum(image_gen_amp_clip_norm*image_gen_clip_norm)/NUM_IMAGES)
                loss+=sourceLoss
            
            ## Id-loss
            IdLoss=torch.zeros(1).cuda()  
            if args.lambda_id:
                IdLoss+=args.lambda_id*id_loss(img_gen_amp0,img_gen0)[0]
                
            loss+=IdLoss

            
            diffs = image_gen_clip_norm - image_gen_amp_clip_norm
            
            
            diffs = diffs / diffs.norm(dim=-1, keepdim=True)
            if args.lambda_consistency > 0:
                diffs_mat_amp = diffs @ diffs.T
                ones_mat = torch.ones(diffs_mat_amp.shape[0]).cuda()
                similarities_loss = torch.sum(ones_mat - diffs_mat_amp) / (NUM_IMAGES ** 2 - NUM_IMAGES)
                loss += args.lambda_consistency * similarities_loss

            loss += target_semantic.reshape(loss.shape)
            reg_loss=torch.zeros(1).cuda()
            
            if args.weight_decay:
                reg_loss+=args.weight_decay*(torch.sum((maskA*At)**2)/2+torch.sum((mask*direction)**2)/2)
                
                
            loss+=reg_loss
            
            avg_coeff=1
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                losslist['loss'][i]=loss.item()
                losslist['tlloss'][i]=target_l.item()
                losslist['closs'][i]=similarities_loss.view(-1).item()*args.lambda_consistency
                losslist['gaploss'][i]=gap_s.item()
                losslist['simloss'][i]=sourceLoss.item()
                
                losslist['idloss'][i]=IdLoss.item()

                losslist['bkloss'][i]=bkloss.item()
                if loss < opt_loss:
                    
                    opt_loss = loss
                    best_direction=direction.clone().detach()
                    best_At=At.clone().detach()
        
            torch.cuda.empty_cache()         

            if i+1==args.step :
                print('********************saving*****************')

                latents = torch.cat(latents_s)

                best_latents=(latents+mask*direction).clone().detach()
                
                latents_0_save=(latents_0+mask*direction).clone().detach()

                torch.save(latents_0_save,args.out_train_latents_path) 

            
                numpy.save('{0}/direction{1}.npy'.format(dir_name, dir_idx),
                                           best_direction.detach().cpu().numpy())
                with torch.no_grad():
                    img_gen, _ = g_ema([latents], input_is_latent=True, randomize_noise=False)
                    direction_all = best_direction 
       
                with torch.no_grad():
                    img_gen_amp, _ = g_ema([best_latents], input_is_latent=True,
                                                               randomize_noise=False)      
                for j in range(latents.shape[0]):
                    torchvision.utils.save_image(img_gen_amp[j],f"{dir_name}/img_gen_amp_{dir_idx}_{j}.png",normalize=True, range=(-1, 1))

        
        numpy.save('{0}/direction{1}{2}.npy'.format(dir_name, dir_idx,args.str),
                               (mask*best_direction).detach().cpu().numpy())


        numpy.save('{0}/At{1}{2}.npy'.format(dir_name, dir_idx,args.str),
                               ((maskA)*best_At+maskAc*(torch.ones(1,18,512).cuda())).detach().cpu().numpy())
            
        torch.save(best_latents,args.out_test_latents_path) 



                      
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_consistency", type=float, default=0.5)
    parser.add_argument("--dir_name", type=str, default=None, help="name of directory to store results")
    parser.add_argument("--output_folder", type=str, default=None, help="path to output folder")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument("--e4e_ckpt", type=str, default="./pretrained_models/e4e_ffhq_encode.pt",
                        help="pretrained e4e weights, in case of initializing from the inversion")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--target_path", type=str, default=None,
                        help="starts the optimization from the given latent code if provided")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")

    parser.add_argument("--lambda_transfer", type=float, default=0)
    
    parser.add_argument("--lambda_gaps", type=float, default=0)

    parser.add_argument("--lambda_id", type=float, default=0)
    
    parser.add_argument("--toL", type=int, default=1, choices=[0,1])

    parser.add_argument("--lambda_background", type=float, default=0)

    parser.add_argument("--lambda_sourcesim", type=float, default=0)
    parser.add_argument("--mpp_d", type=int, default=9)
    
    parser.add_argument("--lambda_transfers", type=float, default=0)
    parser.add_argument("--num_images", type=int, default=4, help="Number of training images")
    parser.add_argument("--test_latents_path", type=str, default="./dirs/latents.npy")
    parser.add_argument("--out_test_latents_path", type=str, default="./dirs/latents_2.pt")
    parser.add_argument("--train_latents_path", type=str, default="./dirs/latents1200.pt")
    parser.add_argument("--out_train_latents_path", type=str, default="./dirs/latents_1200_2.pt")
    

    parser.add_argument("--clip_model", default='ViT-B/16', type=str)
    parser.add_argument("--str", default='', type=str)
    args = parser.parse_args()
    result_image = main(args)