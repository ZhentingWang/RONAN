import torch
from torch.cuda.amp import GradScaler
from inference_utils import SSIMLoss,psnr,lpips_fn,save_img_tensor
from inference_models import get_init_noise, get_model,from_noise_to_image
from inference_image0 import get_image0


import argparse
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
init_noise = get_init_noise(args,args.model_type,bs=args.bs)


init_noise_cal_norm = get_init_noise(args,args.model_type,bs=200)
print("init_noise_cal_norm.shape:",init_noise_cal_norm.shape)
norm_list = []
for i in range(200):
    norm = init_noise_cal_norm[i].norm()
    print("norm:",norm)
    norm_list.append(norm)
avg_norm = sum(norm_list)/len(norm_list)
print("avg_norm:",avg_norm)
#print(init_noise_cal_norm.norm(-1).norm(-1).norm(-1).mean())

#print(gt_noise)
#init_noise = init_noise[0].unsqueeze(0)
#init_noise = gt_noise + torch.randn([1, args.cur_model.z_dim]).cuda()
#exit()

if args.model_type in ["sd"]:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
elif args.model_type in ["sd_unet"]:
    args.cur_model.unet.eval()
    args.cur_model.vae.eval()
    cur_noise_0 = torch.nn.Parameter(torch.tensor(init_noise[0])).cuda()
    #cur_noise_1 = torch.nn.Parameter(torch.tensor(init_noise[1])).cuda()
    optimizer = torch.optim.Adam([cur_noise_0], lr=args.lr)
    #cur_noise_1.requires_grad = False
else:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
    #optimizer = torch.optim.RAdam([cur_noise], lr=args.lr)
    #optimizer = torch.optim.SGD([cur_noise], lr=args.lr, momentum=0.9)

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

#step_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=0.0004)
#step_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1500, gamma=0.5, last_epoch=-1)

import time

args.measure = 9999

if args.mixed_precision:
    scaler = GradScaler()
for i in range(args.num_iter):
    start_time = time.time()
    print("step:",i)
    if args.model_type in ["sd_unet"]:
        cur_noise=[cur_noise_0,cur_noise_1]

    print("cur_noise.shape:",cur_noise.shape)

    if args.mixed_precision:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
            loss = criterion(image0,image).mean()
    else:
        image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
        loss = criterion(image0.detach(),image).mean()

    epoch_num_str=""
    if i%100==0:
        epoch_num_str=str(i)

    with torch.no_grad():
        save_img_tensor(image,"./imgs/image_cur_"+args.input_selection+"_"+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")

    print(criterion(image0,image).mean(-1).mean(-1).mean(-1))
    min_value = criterion(image0,image).mean(-1).mean(-1).mean(-1).min()
    mean_value = criterion(image0,image).mean()
    print("min: ",min_value)
    print("mean: ",mean_value)

    if (args.strategy == "min") and (min_value < args.measure):
        args.measure = min_value
    if (args.strategy == "mean") and (mean_value < args.measure):
        args.measure = mean_value
    print("measure now:",args.measure)

    if args.distance_metric == "lpips":
        loss = loss.mean()
    print("loss "+args.input_selection+" "+args.distance_metric+":",loss)

    if args.mixed_precision:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #step_schedule.step()
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #step_schedule.step()

    #print(gt_noise)
    if gt_noise is not None:
        noise_distance = torch.nn.MSELoss(reduction='none')(cur_noise,gt_noise)
        #print(gt_noise.shape)
        #print(list(range(1,len(gt_noise.shape))))
        print("gt_noise.norm():",gt_noise[0].norm())
        print("noise_distance L2:",noise_distance.mean(-1).mean(-1).mean(-1))
        #print("noise_distance L2:",noise_distance.mean(-1))
        print("cur_noise.norm():",cur_noise[0].norm())

    end_time = time.time()
    print("time for one iter: ",end_time-start_time)
    torch.cuda.empty_cache()

if args.write_txt_path:
    with open(args.write_txt_path,"a") as f:
        f.write(str(args.measure.item())+"\n")

if args.sd_prompt:
    save_img_tensor(image0,"./imgs/ORI_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./imgs/last_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
if args.input_selection_url:
    save_img_tensor(image0,"./imgs/ORI_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./imgs/last_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
