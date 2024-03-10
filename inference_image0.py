
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import random
from urllib.request import urlopen
from nltk.tokenize import RegexpTokenizer
import os
from inference_utils import save_img_tensor
from inference_models import get_init_noise, get_model,from_noise_to_image
import pilgram
def text2img_get_init_image(args):
    
    if args.model_type in ["sd"]:
        if args.sd_prompt:
            prompt = args.sd_prompt
        else:
            prompt = "A cute shiba on the grass"
        image,latents = args.cur_model(prompt, num_inference_steps=50, guidance_scale=7.5,get_latents=True)
        image = image.images[0]
        image = transforms.PILToTensor()(image).cuda()/255
    
    return image,latents

def get_image0(args):
    gt_noise = None
    if args.input_selection == "use_stl10_image0":
        stl10_np = np.load("./data/stl10/train.npz")['x']
        args.dataset_index = random.randint(0, 4999)
        if args.dataset_index:
            stl10_img = stl10_np[args.dataset_index]
            stl10_img_show = Image.fromarray(stl10_img)
        else:
            stl10_img = stl10_np[5]
            stl10_img_show = Image.fromarray(stl10_img)
            stl10_img_show.save("stl10_img_show.jpg")
        image0 = stl10_img

    if args.input_selection == "use_cifar10_image0":
        cifar10_np = np.load("./data/cifar10/train.npz")['x']
        if args.model_type == "styleganv2ada_cifar10":
            cifar10_np_y = np.load("./data/cifar10/train.npz")['y']
            cifar10_class_index_list = [[] for j in range(10)]
            for index in range(cifar10_np.shape[0]):
                cifar10_class_index_list[cifar10_np_y[index][0]].append(index)
            rnd_idx = random.randint(0, len(cifar10_class_index_list[args.stylegan_class_idx]))
            args.dataset_index = cifar10_class_index_list[args.stylegan_class_idx][rnd_idx]
        else:
            args.dataset_index = random.randint(0, 49999)

        if args.dataset_index:
            cifar10_img = cifar10_np[args.dataset_index]
            cifar10_img_show = Image.fromarray(cifar10_img)
        else:
            cifar10_img = cifar10_np[5]
            cifar10_img_show = Image.fromarray(cifar10_img)
            cifar10_img_show.save("cifar10_img_show.jpg")
        cifar10_img = cifar10_img/255
        cifar10_img = torch.from_numpy(cifar10_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = cifar10_img

    if args.input_selection == "use_imagenet_image0":
        imagenet_dir = "./data/imagenet/train/"
        class_dir_list = os.listdir(imagenet_dir)
        rnd_index = random.randint(0,len(class_dir_list)-1)
        class_dir = imagenet_dir+class_dir_list[rnd_index]+"/"
        png_list = os.listdir(class_dir)
        rnd_index = random.randint(0,len(png_list)-1)
        png_file = class_dir+png_list[rnd_index]
        imagenet_img = cv2.imread(png_file)
        b,g,r = cv2.split(imagenet_img)
        imagenet_img = cv2.merge([r, g, b])
        imagenet_img = imagenet_img/255
        imagenet_img = torch.from_numpy(imagenet_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = imagenet_img
        save_img_tensor(image0,"image0_imagenet.png")

    if args.input_selection == "use_generated_image0":
        with torch.no_grad():
            if args.model_type in ["sd"]:
                image0,gt_noise = text2img_get_init_image(args)
            elif "cm" in args.model_type:
                gt_noise = get_init_noise(args,args.model_type,args.cur_model,bs=args.bs)[0].unsqueeze(0).repeat(args.bs,1,1,1)
                image0 = from_noise_to_image(args,args.cur_model,gt_noise,args.model_type)
                save_img_tensor(image0,"image0_cm_samenoise.png")
            else:
                gt_noise = get_init_noise(args,args.model_type,args.cur_model)[0].unsqueeze(0)
                image0 = from_noise_to_image(args,args.cur_model,gt_noise,args.model_type)
            
            save_img_tensor(image0,"image0.png")

    if args.input_selection_model_type != None:
        another_model = get_model(args.input_selection_model_type,args.input_selection_model_path,args)
        with torch.no_grad():
            if "cm" in args.input_selection_model_type:
                another_model_noise = get_init_noise(args,args.input_selection_model_type,another_model,bs=args.bs)
                image0 = from_noise_to_image(args,another_model,another_model_noise,args.input_selection_model_type)[0]
            else:
                another_model_noise = get_init_noise(args,args.input_selection_model_type,another_model,bs=1)
                image0 = from_noise_to_image(args,another_model,another_model_noise,args.input_selection_model_type)

            gt_noise = another_model_noise

    if args.input_selection_url != None:
        readFlag = cv2.IMREAD_COLOR
        resp = urlopen(args.input_selection_url)
        url_img = np.asarray(bytearray(resp.read()), dtype="uint8")
        url_img = cv2.imdecode(url_img, readFlag)

        b,g,r = cv2.split(url_img)
        url_img = cv2.merge([r, g, b])
        url_img = cv2.resize(url_img, (512,512), interpolation=cv2.INTER_AREA)
        url_img_show = Image.fromarray(url_img)
        url_img_show.save("url_img_show.jpg")
        url_img = url_img/255
        url_img = torch.from_numpy(url_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = url_img

    if args.input_selection_name != None:
        shiba_img = cv2.imread(args.input_selection_name)
        b,g,r = cv2.split(shiba_img)
        shiba_img = cv2.merge([r, g, b])
        shiba_img = cv2.resize(shiba_img, (512,512), interpolation=cv2.INTER_AREA)
        #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
        shiba_img_show = Image.fromarray(shiba_img)
        shiba_img_show.save("input_selection_name_img_show3.jpg")
        shiba_img = shiba_img/255
        shiba_img = torch.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = shiba_img

    if args.input_selection != "use_generated_image0" and args.model_type in ["sd"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        image0 = transforms.Resize(height)(image0)
        save_img_tensor(image0,"image0_sd_not_generated.png")

    if args.model_type == "ddpm_cifar10":
        imsize = 32
    elif args.model_type == "dcgan_cifar10":
        imsize = 32
    elif args.model_type == "styleganv2ada_cifar10":
        imsize = 32
    elif args.model_type == "vae_cifar10":
        imsize = 32
    elif "cm" in args.model_type:
        imsize = 64
    elif args.model_type in ["sd"]:
        imsize = 512
    
    image0 = transforms.Resize((imsize,imsize))(image0)

    save_img_tensor(image0,"image0_final.png")

    return image0, gt_noise




