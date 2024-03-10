from cm.script_util import NUM_CLASSES
from cm.random_util import get_generator
from cm.karras_diffusion import get_sigmas_karras,sample_heun,sample_dpm,sample_euler_ancestral,sample_onestep,sample_progdist,sample_euler,stochastic_iterative_sampler
import torch


def cm_inference(args_model_diffusion_tuple,noise):

    cm_args = args_model_diffusion_tuple[0]
    cm_model = args_model_diffusion_tuple[1]
    cm_diffusion = args_model_diffusion_tuple[2]

    def denoiser(x_t, sigma):
        _, denoised = cm_diffusion.denoise(cm_model, x_t, sigma, **cm_args["model_kwargs"])
        if cm_args["clip_denoised"]:
            denoised = denoised.clamp(-1, 1)
        return denoised

    model_kwargs = {}
    if cm_args["class_cond"]:
        '''classes = torch.randint(
            low=0, high=NUM_CLASSES, size=(cm_args["batch_size"],), device="cuda"
        )'''
        classes = torch.tensor([0]*cm_args["batch_size"], dtype=torch.int).cuda()
        model_kwargs["y"] = classes
    cm_args["model_kwargs"] = model_kwargs

    generator=cm_args["generator_"]
    if generator is None:
        generator = get_generator("dummy")
    
    shape = (cm_args["batch_size"], 3, cm_args["image_size"], cm_args["image_size"])
    steps=cm_args["steps"]
    model_kwargs=cm_args["model_kwargs"]
    device="cuda"
    clip_denoised=cm_args["clip_denoised"]
    sampler=cm_args["sampler"]
    sigma_min=cm_args["sigma_min"]
    sigma_max=cm_args["sigma_max"]
    s_churn=cm_args["s_churn"]
    s_tmin=cm_args["s_tmin"]
    s_tmax=cm_args["s_tmax"]
    s_noise=cm_args["s_noise"]
    ts=cm_args["ts_"]
    progress=False,
    callback=None,
    rho=7.0

    steps=cm_args["steps"]
    sigma_min=cm_args["sigma_min"]
    sigma_max=cm_args["sigma_max"]
    s_churn=cm_args["s_churn"]
    s_tmin=cm_args["s_tmin"]
    s_tmax=cm_args["s_tmax"]
    s_noise=cm_args["s_noise"]
    device="cuda"
    sampler=cm_args["sampler"]
    clip_denoised=cm_args["clip_denoised"]

    ts=cm_args["ts_"]
    generator=cm_args["generator_"]
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = noise * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=cm_diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )

    sample = x_0.clamp(-1, 1)
    image = ((sample + 1) * 0.5).clamp(0, 1)

    return image
