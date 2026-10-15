import json, glob
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

cfg_path = glob.glob('/Users/jup33q/.cache/huggingface/hub/models--layerdifforg--seethroughv0.0.2_layerdiff3d/snapshots/*/scheduler/scheduler_config.json')[0]
cfg = json.load(open(cfg_path))
s = DPMSolverMultistepScheduler.from_config(cfg)
s.set_timesteps(30)
ref = {
  "config": cfg,
  "num_inference_steps": 30,
  "timesteps": [int(t) for t in s.timesteps.tolist()],
  "sigmas": [float(v) for v in s.sigmas.tolist()],
  "init_noise_sigma": float(s.init_noise_sigma),
  "scale_factors": [1.0]*30,
}
x = torch.zeros(2,3)
for i,t in enumerate(s.timesteps.tolist()):
    y = s.scale_model_input(x, t)
    assert y is x or torch.equal(y, x)
json.dump(ref, open('/Users/jup33q/see-through/migration/m3/swift/Fixtures/schedule_ref.json','w'), indent=1)
print("sigmas[0..3], [-2:]:", ref["sigmas"][:3], ref["sigmas"][-2:])

# step-replay fixture: deterministic, small shape, explicit variance_noise
rng = np.random.RandomState(0)
shape = (1,2,4,4,4)
sample0 = rng.randn(*shape).astype(np.float32)
mo, vn = [], []
s2 = DPMSolverMultistepScheduler.from_config(cfg)
s2.set_timesteps(30)
lat = sample0.copy()
lat_next = []
for i, t in enumerate(s2.timesteps.tolist()):
    m = rng.randn(*shape).astype(np.float32)
    n = rng.randn(*shape).astype(np.float32)
    mo.append(m.ravel().tolist()); vn.append(n.ravel().tolist())
    out = s2.step(torch.from_numpy(m), t, torch.from_numpy(lat), variance_noise=torch.from_numpy(n), return_dict=False)[0]
    lat = out.numpy().astype(np.float32)
    lat_next.append(lat.ravel().tolist())
steps_ref = {"shape": list(shape), "sample0": sample0.ravel().tolist(),
             "timesteps": ref["timesteps"], "model_outputs": mo, "variance_noises": vn,
             "latents_next": lat_next}
json.dump(steps_ref, open('/Users/jup33q/see-through/migration/m3/swift/Fixtures/steps_ref.json','w'))
print("steps_ref written, final latents[0:4]:", lat_next[-1][:4])
