import json, glob
import numpy as np
import torch
from diffusers import DDIMScheduler

# marigold scheduler config（HF 缓存；HF_HUB_OFFLINE=1）
cfg_path = glob.glob('/Users/jup33q/.cache/huggingface/hub/models--24yearsold--seethroughv0.0.1_marigold/snapshots/*/scheduler/scheduler_config.json')[0]
cfg = json.load(open(cfg_path))
assert cfg['prediction_type'] == 'v_prediction' and cfg['rescale_betas_zero_snr'] and cfg['timestep_spacing'] == 'trailing'
s = DDIMScheduler.from_config(cfg)
s.set_timesteps(4)
ref = {
  "config": cfg,
  "num_inference_steps": 4,
  "timesteps": [int(t) for t in s.timesteps.tolist()],
  "alphas_cumprod": [float(v) for v in s.alphas_cumprod.tolist()],
  "alphas_cumprod_at": {str(t): float(s.alphas_cumprod[t]) for t in [999, 749, 499, 249, 0]},
  "final_alpha_cumprod": float(s.final_alpha_cumprod),
}
json.dump(ref, open('/Users/jup33q/see-through/migration/m3/swift/Fixtures/ddim_schedule_ref.json', 'w'), indent=1)
print("timesteps:", ref["timesteps"], "final_alpha_cumprod:", ref["final_alpha_cumprod"])
print("alphas_cumprod[999], [0]:", ref["alphas_cumprod"][999], ref["alphas_cumprod"][0])

# step-replay fixture：确定性（eta=0 无噪声），小 shape，fp32 CPU
rng = np.random.RandomState(0)
shape = (2, 4, 8, 8)
sample0 = rng.randn(*shape).astype(np.float32)
s2 = DDIMScheduler.from_config(cfg)
s2.set_timesteps(4)
lat = sample0.copy()
mo, lat_next = [], []
for t in s2.timesteps.tolist():
    m = rng.randn(*shape).astype(np.float32)
    mo.append(m.ravel().tolist())
    out = s2.step(torch.from_numpy(m), t, torch.from_numpy(lat), eta=0.0, return_dict=False)[0]
    lat = out.numpy().astype(np.float32)
    lat_next.append(lat.ravel().tolist())
steps_ref = {"shape": list(shape), "sample0": sample0.ravel().tolist(),
             "timesteps": ref["timesteps"], "model_outputs": mo,
             "latents_next": lat_next}
json.dump(steps_ref, open('/Users/jup33q/see-through/migration/m3/swift/Fixtures/ddim_steps_ref.json', 'w'))
print("steps_ref written, final latents[0:4]:", lat_next[-1][:4])
