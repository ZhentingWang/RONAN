import torch
from torch.cuda.amp import GradScaler
from inference_utils import SSIMLoss,psnr,lpips_fn,save_img_tensor
from inference_models import get_init_noise, get_model,from_noise_to_image
from inference_image0 import get_image0
import argparse
import numpy as np
import complexity
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
parser.add_argument("--distance_metric", default="l2", type=str, help="The path of dev set.")
parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

parser.add_argument("--lr", default=1e-2, type=float, help="")
parser.add_argument("--dataset_index", default=None, type=int, help="")
parser.add_argument("--bs", default=8, type=int, help="")
parser.add_argument("--write_txt_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
parser.add_argument("--strategy", default="mean", type=str, help="The path of dev set.")
parser.add_argument("--mixed_precision", action="store_true", help="The path of dev set.")
parser.add_argument("--sd_prompt", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")

args = parser.parse_args()

args.cur_model = get_model(args.model_type,args.model_path_,args)
image0, gt_noise = get_image0(args)
image0 = image0.detach()
init_noise = get_init_noise(args,args.model_type,args.cur_model,bs=args.bs)

if args.model_type in ["sd"]:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
elif args.model_type in ["sd_unet"]:
    args.cur_model.unet.eval()
    args.cur_model.vae.eval()
    cur_noise_0 = torch.nn.Parameter(torch.tensor(init_noise[0])).cuda()
    optimizer = torch.optim.Adam([cur_noise_0], lr=args.lr)
else:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
    
if args.distance_metric == "l1":
    criterion = torch.nn.L1Loss(reduction='none')
elif args.distance_metric == "l2":
    criterion = torch.nn.MSELoss(reduction='none')
elif args.distance_metric == "ssim":
    criterion = SSIMLoss().cuda()
elif args.distance_metric == "psnr":
    criterion = psnr
elif args.distance_metric == "lpips":
    criterion = lpips_fn
    
import time
args.measure = float("inf")

if args.mixed_precision:
    scaler = GradScaler()
for i in range(args.num_iter):
    start_time = time.time()
    print("step:",i)

    if args.mixed_precision:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
            loss = criterion(image0,image).mean()
    else:
        image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
        loss = criterion(image0.detach(),image).mean()

    if i%100==0:
        epoch_num_str=str(i)
        with torch.no_grad():
            save_img_tensor(image,"./result_imgs/image_cur_"+args.input_selection+"_"+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")

    min_value = criterion(image0,image).mean(-1).mean(-1).mean(-1).min()
    mean_value = criterion(image0,image).mean()

    if (args.strategy == "min") and (min_value < args.measure):
        args.measure = min_value
    if (args.strategy == "mean") and (mean_value < args.measure):
        args.measure = mean_value
    print("lowest loss now:",args.measure.item())

    if args.distance_metric == "lpips":
        loss = loss.mean()
    print("loss "+args.input_selection+" "+args.distance_metric+":",loss.item())

    if args.mixed_precision:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print("time for one iter: ",end_time-start_time)
    torch.cuda.empty_cache()


cv2_img0 = (image0.squeeze(0).permute(1, 2, 0).cpu().numpy()* 255).astype(np.uint8)
cv2_img0 = cv2.cvtColor(cv2_img0, cv2.COLOR_BGR2GRAY)

print("*"*80)
print("final lowest loss: ",args.measure.item())
print("2D-entropy-based complexity: ", complexity.calcEntropy2dSpeedUp(cv2_img0, 3, 3))

if args.write_txt_path:
    with open(args.write_txt_path,"a") as f:
        f.write(str(args.measure.item())+"\n")

if args.sd_prompt:
    save_img_tensor(image0,"./result_imgs/ORI_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./result_imgs/last_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
if args.input_selection_url:
    save_img_tensor(image0,"./result_imgs/ORI_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./result_imgs/last_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
