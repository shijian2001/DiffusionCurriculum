import contextlib
import datetime
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.distributions.kl as kl
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.loaders import AttnProcsLayers
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from transformers import Pipeline
from transformers.pipelines import pipeline

from train.curriculum import Curriculum
from train.trainer.common.ddim_with_logprob import ddim_step_with_logprob
from train.trainer.common.pipeline_with_logprob import pipeline_with_logprob
from train.trainer.common.state_tracker import PerPromptStatTracker

t = partial(tqdm.tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    sample_num_steps: int = field(default=50)
    train_num_steps: int = field(default=1000)
    timestep_fraction: float = field(default=0.8)
    log_dir: str = field(default="logs")
    sd_model: str = field(default="runwayml/stable-diffusion-v1-5")
    sd_revision: str = field(default="main")
    learning_rate: float = field(default=1e-4)
    # whether to use LoRA for training instead of full model.
    use_lora: bool = field(default=False)

    # random seed for reproducibility.
    seed: int = field(default=0)
    # top-level logging directory for checkpoint saving.
    logdir: str = field(default="logs")
    # logging platform to report to. Use "wandb" for Weights & Biases, or "none" to disable.
    report_to: str = field(default="wandb")
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    num_epochs: int = field(default=400)
    # number of epochs between saving model checkpoints.
    save_freq: int = field(default=400)
    # number of checkpoints to keep before overwriting old ones.
    num_checkpoint_limit: int = field(default=10)
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    mixed_precision: str = field(default="fp16")
    # allow tf32 on Ampere GPUs, which can speed up training.
    allow_tf32: bool = field(default=True)
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    resume_from: str = field(default="")
    # whether or not to use xFormers to reduce memory usage.
    use_xformers: bool = field(default=False)

    ############ Sampling ############
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample_eta: float = field(default=1.0)
    # classifier-free guidance weight. 1.0 is no guidance.
    sample_guidance_scale: float = field(default=5.0)
    # batch size (per GPU!) to use for sampling.
    sample_batch_size: int = field(default=1)
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample_num_batches_per_epoch: int = field(default=2)
    # save interval
    sample_save_interval: int = field(default=100)

    ############ Training ############
    # batch size (per GPU!) to use for training.
    train_batch_size: int = field(default=1)
    # learning rate.
    train_learning_rate: float = field(default=3e-5)
    # Adam beta1.
    adam_beta1: float = field(default=0.9)
    # Adam beta2.
    adam_beta2: float = field(default=0.999)
    # Adam weight decay.
    adam_weight_decay: float = field(default=1e-4)
    # Adam epsilon.
    adam_epsilon: float = field(default=1e-8)
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    gradient_accumulation_steps: int = field(default=1)
    # maximum gradient norm for gradient clipping.
    train_max_grad_norm: float = field(default=1.0)
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    num_inner_epochs: int = field(default=1)
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train_cfg: bool = field(default=True)
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train_adv_clip_max: float = field(default=5)
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train_timestep_fraction: float = field(default=1.0)
    # DDPO: the PPO clip range.
    train_clip_range: float = field(default=1e-4)
    # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
    # advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
    # advantages will be calculated using the mean and std of the entire batch.
    per_prompt_stat_tracking: bool = field(default=True)
    # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    per_prompt_stat_tracking_buffer_size: int = field(default=16)
    # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
    # contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    per_prompt_stat_tracking_min_count: int = field(default=16)

    # DPOK: the KL coefficient
    kl_ratio: float = field(default=0.01)

    ############ Prompt Function ############
    # prompt function to use. see `prompts.py` for available prompt functisons.
    prompt_fn: str = field(default="simple_animals")

    ############ Reward Function ############
    # reward function to use. see `rewards.py` for available reward functions.
    # if the reward_fn is "jpeg_compressibility" or "jpeg_incompressibility", using the default config can reproduce our results.
    # if the reward_fn is "aesthetic_score" and you want to reproduce our results,
    # set config.num_epochs = 1000, sample.num_batches_per_epoch=1, sample.batch_size=8 and sample.eval_batch_size=8
    reward_fn: str = field(default="jpeg_compressibility")


class Trainer:
    def __init__(
        self,
        curriculum: Curriculum,
        update_target_difficulty: Callable[[int], None],
        config: Config,
        reward_function: Callable[[Pipeline, torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        reward_init_function: Callable[[Accelerator, int], None],
        prompt_function: Callable[[], tuple[str, Any]],
        vqa_model_name: str,
    ) -> None:
        self.curriculum = curriculum
        self.update_target_difficulty = update_target_difficulty
        self.config = config
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.run_name = f"dpok_{unique_id}"
        if self.config.resume_from:
            self.config.resume_from = self._norm_path(self.config.resume_from)

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.timestep_fraction)

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.log_dir, self.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,
        )
        self.stat_tracker = None
        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        self.vqa_pipeline = pipeline(
            "image-text-to-text",
            model=vqa_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            batch_size=config.train_batch_size,
        )

        log_with = None if self.config.report_to.lower() == "none" else self.config.report_to

        self.accelerator = Accelerator(
            log_with=log_with,
            project_config=accelerator_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.gradient_accumulation_steps * self.num_train_timesteps,
        )
        reward_init_function(self.accelerator, self.config.sample_batch_size)
        self.available_devices = self.accelerator.num_processes
        self._fix_seed()
        if self.accelerator.is_main_process and self.config.report_to.lower() != "none":
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch",
                config=asdict(self.config),
                init_kwargs={"wandb": {"name": self.run_name}},
            )
        logger.info(f"\n{self.config}")

        # load scheduler, tokenizer and models.
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model, revision=self.config.sd_revision
        )
        # freeze parameters of models to save more memory
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(not self.config.use_lora)
        # disable safety checker
        self.sd_pipeline.safety_checker = None
        # make the progress bar nicer
        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # switch to DDIM scheduler
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to inference_dtype
        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        if self.config.use_lora:
            self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        if self.config.use_lora:
            unet_lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )

            self.sd_pipeline.unet.add_adapter(unet_lora_config)

            for param in self.sd_pipeline.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)

        trainable_layers = self.sd_pipeline.unet

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        self.optimizer = torch.optim.AdamW(
            trainable_layers.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        self.prompt_fn = prompt_function
        self.reward_fn = reward_function

        # generate negative prompt embeddings
        neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample_batch_size, 1, 1)
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.train_batch_size, 1, 1)

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = contextlib.nullcontext if self.config.use_lora else self.accelerator.autocast

        # Prepare everything with our `accelerator`.
        trainable_layers, optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        self.optimizer = optimizer

        self.samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        )

        assert self.config.sample_batch_size >= self.config.train_batch_size
        assert self.config.sample_batch_size % self.config.train_batch_size == 0
        assert self.samples_per_epoch % self.total_train_batch_size == 0

        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)
            self.first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def _fix_seed(self):
        assert self.accelerator, "should call after init accelerator"
        # set seed (device_specific is very important to get different prompts on different devices)
        np.random.seed(self.config.seed or 114514)
        random_seeds = np.random.randint(0, 100000, size=self.available_devices)
        device_seed = random_seeds[self.accelerator.process_index]  # type: ignore
        set_seed(int(device_seed), device_specific=True)

    def _norm_path(self, path: str) -> str:
        res = os.path.normpath(os.path.expanduser(path))
        if "checkpoint_" not in os.path.basename(path):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(self.config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {path}")
            res = os.path.join(
                path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        return res

    def _save_model_hook(self, models, weights, output_dir):
        assert len(models) == 1
        if self.config.use_lora:
            unwrapped_unet = self.accelerator.unwrap_model(models[0])
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

            self.sd_pipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
        elif isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            # sd_pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                self.config.sd_model, revision=self.config.sd_revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)  # type: ignore
            models[0].load_state_dict(load_model.state_dict())  # type: ignore
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.config.num_epochs}")
        logger.info(f"  Sample batch size per device = {self.config.sample_batch_size}")
        logger.info(f"  Train batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info("")
        logger.info(f"  Total number of samples per epoch = {self.samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = {self.samples_per_epoch // self.total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {self.config.num_inner_epochs}")
        global_step = 0
        for epoch in range(self.first_epoch, self.config.num_epochs):
            global_step = self.epoch_loop(global_step, epoch)

    def _sample(self, epoch: int, global_step: int):
        samples: list[dict] = []
        prompts = []
        for _ in t(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(self.config.sample_batch_size)])

            # encode prompts
            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            # sample
            with self.autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    self.sd_pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # 直接计算奖励
            rewards, reward_metadata = self.reward_fn(self.vqa_pipeline, images, prompts, prompt_metadata)
            rewards = torch.as_tensor(rewards, device=self.accelerator.device)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        zip_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = self.accelerator.gather(zip_samples["rewards"]).cpu().numpy()

        self.accelerator.log(
            {
                "reward": rewards,
                "num_samples": epoch * self.available_devices * self.config.sample_batch_size,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.to(torch.float).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            self.accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                    ],
                },
                step=global_step,
            )

        return zip_samples, prompts, rewards

    def epoch_loop(self, global_step: int, epoch: int):
        #################### SAMPLING ####################
        self.sd_pipeline.unet.eval()

        samples, prompts, rewards = self._sample(epoch, global_step)

        # per-prompt mean/std tracking
        if self.stat_tracker:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        assert num_timesteps == self.config.sample_num_steps

        #################### TRAINING ####################
        for inner_epoch in range(self.config.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None], perms
                ]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            self.sd_pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in t(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                self.step(sample, i, epoch, inner_epoch, global_step, info)
                global_step += 1

            # make sure we did an optimization step at the end of the inner epoch
            assert self.accelerator.sync_gradients

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step

    def step(self, sample: dict, step: int, epoch: int, inner_epoch: int, global_step: int, info: dict):
        if self.config.train_cfg:
            # concat negative prompts to sample prompts to avoid two forward passes
            embeds = torch.cat([self.train_neg_prompt_embeds, sample["prompt_embeds"]])
        else:
            embeds = sample["prompt_embeds"]

        for j in t(
            range(self.num_train_timesteps),
            desc="Timestep",
            position=1,
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        ):
            with self.accelerator.accumulate(self.sd_pipeline.unet):
                with self.autocast():
                    if self.config.train_cfg:
                        noise_pred = self.sd_pipeline.unet(
                            torch.cat([sample["latents"][:, j]] * 2),
                            torch.cat([sample["timesteps"][:, j]] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = self.sd_pipeline.unet(
                            sample["latents"][:, j], sample["timesteps"][:, j], embeds
                        ).sample

                    # compute the log prob of next_latents given latents under the current model
                    _, log_prob = ddim_step_with_logprob(
                        self.sd_pipeline.scheduler,
                        noise_pred,
                        sample["timesteps"][:, j],
                        sample["latents"][:, j],
                        eta=self.config.sample_eta,
                        prev_sample=sample["next_latents"][:, j],
                    )

                # ppo logic
                advantages = torch.clamp(
                    sample["advantages"], -self.config.train_adv_clip_max, self.config.train_adv_clip_max
                )
                ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio, 1.0 - self.config.train_clip_range, 1.0 + self.config.train_clip_range
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                kl_divergence = kl.kl_divergence(
                    torch.distributions.Categorical(logits=log_prob),
                    torch.distributions.Categorical(logits=sample["log_probs"][:, j]),
                )
                loss += self.config.kl_ratio * kl_divergence.mean()
                # debugging values
                # John Schulman says that (ratio - 1) - log(ratio) is a better
                # estimator, but most existing code uses this so...
                # http://joschu.net/blog/kl-approx.html
                info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float()))
                info["loss"].append(loss)

                # backward pass
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.sd_pipeline.unet.parameters(), self.config.train_max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                assert (j == self.num_train_timesteps - 1) and (step + 1) % self.config.gradient_accumulation_steps == 0
                # log training-related stuff
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = self.accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                self.accelerator.log(info, step=global_step)
                info = defaultdict(list)

    def _unwrap_model(self, model):
        """Unwraps model from accelerator wrapper, if needed."""
        return self.accelerator.unwrap_model(model)
