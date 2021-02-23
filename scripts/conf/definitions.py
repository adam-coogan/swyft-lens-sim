import os
import pyro
import torch
import numpy as np
from operator import itemgetter
from ruamel.yaml import YAML
from torch import nn

from clipppy.patches import torch_numpy
from clipppy import load_config
import swyft


"""

Prior, model and noise function definitions for lensing analysis.

Running this file will run some basic tests. This should produce a file called
`resids.png` and print out the guide, which should be:

    >>> Guide(
    ...     (gp_alpha): DeltaSamplingGroup(1 sites, torch.Size([2]))
    ...     (gp): DiagonalNormalSamplingGroup(1 sites, torch.Size([56454]))
    ...     (src_alpha): DeltaSamplingGroup(1 sites, torch.Size([1]))
    ...     (g): PartialMultivariateNormalSamplingGroup(8 sites, torch.Size([1607]))
    ... )

"""


def get_config():
    """
    Get the config without polluting the global namespace.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK

    BASE_DIR = "resources"
    SYSTEM_NAME = "hoags_object"
    PARAMS = YAML().load(open(os.path.join(BASE_DIR, "params.yaml")))[SYSTEM_NAME]

    config = load_config(os.path.join(BASE_DIR, "config-sh.yaml"), base_dir=BASE_DIR)
    model = config.umodel

    config.umodel.sources["src"] = config.kwargs["defs"]["src"]
    config.umodel.sources["gp"] = config.kwargs["defs"]["gp"]
    config.umodel.alphas["main"].stochastic_specs["slope"] = PARAMS["truth"][
        "main/slope"
    ]

    config.guide.setup()
    config.guide = torch.load(
        os.path.join(BASE_DIR, f"guide-{SYSTEM_NAME}-fixed-final.pt")
    )

    torch.set_default_tensor_type(torch.FloatTensor)  # HACK

    return config


def get_prior(config):
    """
    Set up subhalo parameter priors using a config.
    """
    main = config.umodel.alphas["main"]
    prior_p_sub = main.sub.pos_sampler.base_dist
    prior_log10_m_sub = main.sub.mass_sampler.base_dist
    return swyft.Prior(
        {
            "x_sub": ["uniform", prior_p_sub.low[0].cpu(), prior_p_sub.high[0].cpu()],
            "y_sub": ["uniform", prior_p_sub.low[1].cpu(), prior_p_sub.high[1].cpu()],
            "log10_m_sub": [
                "uniform",
                prior_log10_m_sub.low.cpu(),
                prior_log10_m_sub.high.cpu(),
            ],
        }
    )


def model(params):
    """
    Sample from the config's PPD, potentially with some parameters fixed, and put
    in a subhalo.

    Requires the global variable `config` to be set.

    Arguments
    - params: dict containing keys "x_sub", "y_sub", "log10_m_sub" whose values can
      be converted to floats.

    Returns
    - Numpy array. Could return a torch.Tensor if that would be more convenient.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK

    x_sub, y_sub, log10_m_sub = itemgetter("x_sub", "y_sub", "log10_m_sub")(
        {k: float(v) for k, v in params.items()}
    )
    d_m_sub = dist.Delta(torch.tensor(10 ** log10_m_sub))
    d_p_sub = dist.Delta(torch.tensor([x_sub, y_sub])).to_event(1)

    def _guide():
        # Sample subhalo guide
        m_sub = pyro.sample("main/sub/m_sub", d_m_sub)
        p_sub = pyro.sample("main/sub/p_sub", d_p_sub)
        # Sample from lens and source-plane GP guide
        config.guide.g()
        # Sample from image-plane GP guide
        config.guide.gp()

    result = {
        "mu": config.ppd(guide=_guide)["model_trace"]
        .nodes["mu"]["value"]
        .detach()
        .numpy()
    }

    torch.set_default_tensor_type(torch.FloatTensor)  # HACK

    return result


def noise(obs, params=None):
    """
    Noise model: adds Gaussian pixel noise to the image with the level specified
    when the config is loaded above.

    Requires the global variable `sigma_n` to be set.
    """
    mu = obs["mu"]
    eps = np.random.randn(*mu.shape) * sigma_n

    return {"mu": mu + eps}


class CustomHead(swyft.Module):
    """
    Head network used for NeurIPS paper (https://arxiv.org/abs/2010.07032). The
    BatchNorm2d layers were key.
    """

    def __init__(self, obs_shapes):
        super().__init__(obs_shapes=obs_shapes)

        self.n_features = 128

        self.layers = nn.Sequential(
            nn.Conv2d(1, 2, 8, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, 8, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 8, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 8, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 8, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 8, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 8, 2, 1),
        )

    def forward(self, obs):
        x = obs["mu"].unsqueeze(1)  # add channel dimension
        nbatch = len(x)
        x = self.layers(x)
        x = x.view(nbatch, -1)
        return x


# Set everything up
config = get_config()
prior = get_prior(config)
sigma_n = config.umodel.stochastic_specs["sigma_stat"]

par0 = {k: float(v) for k, v in prior.sample(1).items()}
obs0 = noise(model(par0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    im_kwargs = dict(extent=(-2.5, 2.5, -2.5, 2.5), origin="lower")
    cbar_kwargs = dict(fraction=0.046, pad=0.04)
    n_ppd = 100

    print("Testing lens model\n")

    print("Head network output:")
    head = CustomHead(config.umodel.X.shape)
    single_sample = {
        k: torch.tensor(v).unsqueeze(0) for k, v in model(prior.sample(1)).items()
    }
    print(head(single_sample))
    print("Shape:", head(single_sample).shape, "\n")

    print("Guide:\n", config.guide, "\n")

    pred = np.stack([model(prior.sample(1))["mu"] for i in range(n_ppd)], 0).mean(0)
    OBS = config.conditioning["image"].numpy()
    MASK = config.kwargs["defs"]["mask"].numpy()
    err = np.ma.array((pred - OBS) / sigma_n, mask=~MASK)
    vm = 2 * np.sqrt((err ** 2).mean())

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    im = axes[0].imshow(OBS / sigma_n, **im_kwargs)
    plt.colorbar(im, ax=axes[0], **cbar_kwargs)
    im = axes[1].imshow(pred / sigma_n, **im_kwargs)
    plt.colorbar(im, ax=axes[1], **cbar_kwargs)
    im = axes[2].imshow(err, **im_kwargs, cmap="bwr", vmin=-vm, vmax=vm)
    plt.colorbar(im, ax=axes[2], **cbar_kwargs)
    plt.tight_layout()
    fig.savefig("resids.png")

    print("Saved test plot to resids.png\n")

    fps = 2
    duration = 10  # s
    shs = [prior.sample(1) for _ in range(duration * fps)]
    snapshots = [model(sh)["mu"] for sh in shs]

    fig = plt.figure(figsize=(4, 4))

    a = snapshots[0]
    im = plt.imshow(
        a,
        interpolation="none",
        aspect="auto",
        vmin=-3,
        vmax=30,
        extent=(-2.5, 2.5, -2.5, 2.5),
        origin="lower",
    )
    scat = plt.scatter(
        shs[0]["x_sub"], shs[0]["y_sub"], s=10 * np.sqrt(shs[0]["log10_m_sub"]), c="r"
    )
    plt.axis("off")
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print(".", end="")
        im.set_array(snapshots[i])
        scat.set_offsets(np.hstack([shs[i]["x_sub"], shs[i]["y_sub"]]))
        scat.set_sizes(10 * np.sqrt(shs[i]["log10_m_sub"]))
        return [im, scat]

    anim = animation.FuncAnimation(
        fig, animate_func, frames=duration * fps, interval=1000 / fps  # ms
    )

    anim.save("samples.gif", fps=fps)

    print("Saved animation to samples.gif.\n")

    print("Done.")
