import os
import socket
import logging
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm

import jax
from jax import numpy as jnp
import optax
import haiku as hk

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from score_sde.models.flow import SDEPushForward
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref, animate_sampling

log = logging.getLogger(__name__)


def _unsafe_wedge_mask(phi, phi_min, phi_max, xp=jnp):
    """Mask for phi in the configured wedge, mirrored to the opposite side of the ring (+pi)."""
    center = (phi_min + phi_max) / 2
    half_width = (phi_max - phi_min) / 2
    center2 = xp.arctan2(xp.sin(center + xp.pi), xp.cos(center + xp.pi))

    def in_wedge(c):
        diff = xp.arctan2(xp.sin(phi - c), xp.cos(phi - c))
        return xp.abs(diff) < half_width

    return in_wedge(center) | in_wedge(center2)


def _unsafe_component_mask(x, center, radius, xp=jnp):
    """Mask for points within geodesic `radius` of a fixed direction `center` on S2."""
    cos_dist = xp.clip(x @ xp.asarray(center), -1.0, 1.0)
    return xp.arccos(cos_dist) < radius


def _get_unsafe_mask_fn(safety, dataset):
    """Build a fn x0 -> bool mask flagging the configured unsafe region.

    region="wedge" (default): azimuthal band [phi_min, phi_max] mirrored at +pi.
    region="component": geodesic cap of radius `cap_radius` around the mean
    direction of Kent mixture component `unsafe_component`.
    """
    region = getattr(safety, "region", "wedge")
    if region == "component":
        base_ds = dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        means = getattr(base_ds, "means", None)
        if means is None:
            raise ValueError(
                "safety.region='component' requires a dataset exposing `.means` "
                "(e.g. KentSynthetic); got "
                f"{type(base_ds).__name__} which has no `.means` attribute."
            )
        if not (0 <= safety.unsafe_component < len(means)):
            raise ValueError(
                f"safety.unsafe_component={safety.unsafe_component} is out of range "
                f"for dataset with {len(means)} components."
            )
        center = jnp.asarray(means[safety.unsafe_component])

        def mask_fn(x0, xp=jnp):
            return _unsafe_component_mask(x0, center, safety.cap_radius, xp=xp)

        return mask_fn
    else:
        def mask_fn(x0, xp=jnp):
            phi = xp.arctan2(x0[:, 1], x0[:, 0])
            return _unsafe_wedge_mask(phi, safety.phi_min, safety.phi_max, xp=xp)

        return mask_fn


def _compute_mmd(x_gen, x_real, n_subsample=2000):
    x = np.array(x_gen[:n_subsample])
    y = np.array(x_real[:n_subsample])

    def geodesic_dists(a, b):
        return np.arccos(np.clip(a @ b.T, -1.0, 1.0))

    bw = float(np.median(geodesic_dists(x[:200], y[:200]).ravel()))
    if bw < 1e-6:
        bw = 1.0

    def k(a, b):
        return np.exp(-geodesic_dists(a, b) ** 2 / (2 * bw ** 2))

    mmd2 = k(x, x).mean() - 2 * k(x, y).mean() + k(y, y).mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def run(cfg):
    def train(train_state):
        loss = instantiate(
            cfg.loss, pushforward=pushforward, model=model, eps=cfg.eps, train=True
        )
        train_step_fn = get_ema_loss_step_fn(loss, optimizer=optimiser, train=True)
        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        t = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()
        total_train_time = 0
        for step in t:
            data, context = next(train_ds)
            batch = {"data": data, "context": context}
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan")
                return train_state, False

            if step % 50 == 0:
                logger.log_metrics({"train/loss": loss}, step)
                t.set_description(f"Loss: {loss:.3f}")

            if step > 0 and step % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time
                save(ckpt_path, train_state)
                eval_time = timer()
                if cfg.train_val:
                    evaluate(train_state, "val", step)
                    logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)
                if cfg.train_plot:
                    generate_plots(train_state, "val", step=step)

                if cfg.train_animate:
                    volcano_data = []
                    for batch in eval_ds:
                        volcano_data.append(batch[0])
                    volcano_data = np.concatenate(volcano_data, axis=0)

                    animate_sampling(
                        pushforward=pushforward,
                        model=model,
                        train_state=train_state,
                        epoch=step,
                        cfg=cfg,
                        volcano_data=volcano_data,
                        save_path=os.path.join(run_path, 'animations')
                    )
                train_time = timer()

        logger.log_metrics({"train/total_time": total_train_time}, step)

        if cfg.train_animate:
            print(f"Generating final animation at step {cfg.steps}...")
            volcano_data, _ = next(eval_ds)
            animate_sampling(
                pushforward=pushforward,
                model=model,
                train_state=train_state,
                epoch=cfg.steps,
                cfg=cfg,
                volcano_data=volcano_data,
                save_path=os.path.join(run_path, 'animations')
            )

        return train_state, True

    def evaluate(train_state, stage, step=None):
        log.info("Running evaluation")
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        likelihood_fn = jax.jit(likelihood_fn)

        logp, nfe, N = 0.0, 0.0, 0
        tot = 0

        if hasattr(dataset, "__len__"):
            for batch in dataset:
                logp_step, nfe_step = likelihood_fn(*batch)
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
        else:
            dataset.batch_dims = [cfg.eval_batch_size]
            samples = round(20_000 / cfg.eval_batch_size)
            for i in range(samples):
                batch = next(dataset)
                logp_step, nfe_step = likelihood_fn(*batch)
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
                tot += logp_step.shape[0]
            dataset.batch_dims = [cfg.batch_size]

        logp /= N
        nfe /= len(dataset) if hasattr(dataset, "__len__") else samples

        logger.log_metrics({f"{stage}/logp": logp}, step)
        log.info(f"{stage}/logp = {logp:.3f}")
        logger.log_metrics({f"{stage}/nfe": nfe}, step)
        log.info(f"{stage}/nfe = {nfe:.1f}")

        if stage == "test":  # Estimate normalisation constant
            default_context = context[0] if context is not None else None
            Z = compute_normalization(
                likelihood_fn, data_manifold, context=default_context
            )
            log.info(f"Z = {Z:.2f}")
            logger.log_metrics({f"{stage}/Z": Z}, step)

    def generate_plots(train_state, stage, step=None):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "val" else test_ds
        
        # Collect unsafe reference points if safety guardrail is enabled
        safety = cfg.safety
        unsafe_mask_fn = _get_unsafe_mask_fn(safety, dataset)
        unsafe_points = None
        if safety.enabled:
            unsafe_points_list = []
            while True:
                x0, _ = next(dataset)
                mask = unsafe_mask_fn(x0)
                selected = x0[mask]
                if selected.shape[0] > 0:
                    unsafe_points_list.append(selected)
                if len(unsafe_points_list) > 0:
                    unsafe_points = jnp.concatenate(unsafe_points_list, axis=0)
                    if unsafe_points.shape[0] >= safety.target_n:
                        unsafe_points = unsafe_points[:safety.target_n]
                        break
            region_desc = (
                f"component {safety.unsafe_component} (cap radius={safety.cap_radius:.2f})"
                if getattr(safety, "region", "wedge") == "component"
                else f"φ∈[{safety.phi_min:.2f},{safety.phi_max:.2f}] and its mirror at +π"
            )
            log.info(f"Safety guardrail enabled: collected {safety.target_n} unsafe points across {region_desc}")

        M = 32 if isinstance(pushforward, SDEPushForward) else 8
        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler_kwargs = dict(
            N=100,
            eps=cfg.eps,
            predictor="GRW",
        )

        # Naive noise rejection: forward-diffuse unsafe points to noise space once
        method = getattr(safety, 'method', 'early_window')
        noised_unsafe_points = None
        if safety.enabled and unsafe_points is not None and method == 'noise_rejection':
            rng, fwd_rng = jax.random.split(rng)
            noised_unsafe_points = pushforward.sde.marginal_sample(fwd_rng, unsafe_points, t=pushforward.sde.tf)
            log.info(f"Forward diffused {unsafe_points.shape[0]} unsafe points to t={pushforward.sde.tf:.2f} (noise space)")

        sampler = pushforward.get_sampler(
            model_w_dicts,
            train=False,
            unsafe_points=unsafe_points if method in ['early_window', 'full_window_scaled', 'late_window_scaled'] else None,
            safety_cfg=safety if safety.enabled else None,
            noised_unsafe_points=noised_unsafe_points,
            **sampler_kwargs)

        x0, context = next(dataset)
        shape = (int(cfg.batch_size * M),)
        rng, next_rng = jax.random.split(rng)
        x = sampler(next_rng, shape, context)
        prop_in_M = data_manifold.belongs(x, atol=1e-4).mean()
        log.info(f"Prop samples in M = {100 * prop_in_M.item():.1f}%")

        asr = jnp.mean(unsafe_mask_fn(x)).item()
        log.info(f"Proportion of generated samples in unsafe region = {100 * asr:.2f}%")
        logger.log_metrics({"safety/asr": asr}, step)

        # --- MMD between generated and real test samples ---
        real_batches = []
        n_real_target = 2000
        while sum(b.shape[0] for b in real_batches) < n_real_target:
            real_batches.append(np.array(next(dataset)[0]))
        x_real = np.concatenate(real_batches, axis=0)
        x_for_mmd = np.array(x)
        mmd = _compute_mmd(x_for_mmd, x_real)
        log.info(f"{stage}/mmd = {mmd:.4f}")
        logger.log_metrics({f"{stage}/mmd": mmd}, step)

        # Safe MMD: real reference restricted to the safe region, computed regardless
        # of whether the safety mechanism is on, so baseline runs are comparable.
        phi_real = np.arctan2(x_real[:, 1], x_real[:, 0])
        # x_real_safe = x_real[~_unsafe_wedge_mask(phi_real, safety.phi_min, safety.phi_max, xp=np)]
        # x_real_safe = x_real[~_unsafe_cap_mask(phi_real, safety.phi_min, safety.phi_max, xp=np)]
        # log.info(f"Safe MMD reference: {x_real_safe.shape[0]} safe real samples")
        # safe_mmd = _compute_mmd(x_for_mmd, x_real_safe)
        # log.info(f"{stage}/safe_mmd = {safe_mmd:.4f}")
        # logger.log_metrics({f"{stage}/safe_mmd": safe_mmd}, step)

        # --- samples from model (original plot, no vectors) ---
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        log_prob = jax.jit(lambda x: likelihood_fn(x)[0])
        unsafe_label = (
            f"unsafe (η={safety.eta}, t_min={safety.t_min})"
            if safety.enabled else "unsafe"
        )
        plt = plot(data_manifold, None, x, log_prob=log_prob, unsafe_points=unsafe_points, unsafe_label=unsafe_label)
        logger.log_plot("x0_bwd", plt, step)

        # --- vector field on uniform grid ---
        rng, next_rng = jax.random.split(rng)
        n_grid = 500
        grid_points = jnp.array(data_manifold.random_uniform(state=rng, n_samples=n_grid))
        t_array = jnp.full((n_grid, 1), cfg.eps)
        vectors, _ = model.apply(
            train_state.params_ema,
            train_state.model_state,
            next_rng,
            y=grid_points,
            t=t_array,
            context=None,
        )
        vectors = np.array(vectors)
        grid_points = np.array(grid_points)
        fig = plot(data_manifold, None, None, log_prob=None, vectors=vectors, vector_origins=grid_points)
        logger.log_plot("vector_field", fig, step)

        # --- samples from data (only at step 0) ---
        if step <= 0:
            dataset.batch_dims = shape[0]
            x0 = next(dataset)[0]
            log_prob_data = dataset.log_prob if hasattr(dataset, "log_prob") else None
            plt = plot(data_manifold, None, x0, log_prob=log_prob_data)
            logger.log_plot("x0", plt, step)
            dataset.batch_dims = cfg.batch_size

        # --- forward process (only at step 0) ---
        if step <= 0 and isinstance(pushforward, SDEPushForward):
            sampler = pushforward.get_sampler(
                model_w_dicts, train=False, reverse=False, **sampler_kwargs
            )
            zT = sampler(rng, None, context, z=transform.inv(x0))
            plt = plot_ref(model_manifold, transform.inv(zT), log_prob=base.log_prob)
            logger.log_plot("xT_fwd", plt, step)

    ### Main
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info("Stage : Instantiate model")
    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    log.info("Stage : Instantiate dataset")
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    log.info("Stage : Instantiate vector field model")

    def model(y, t, context=None):
        """Vector field s_\theta: y, t, context -> T_y M"""
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        # TODO: parse context into embedding map
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size, 1))
    data, context = next(train_ds)
    params, state = model.init(rng=next_rng, y=transform.inv(data), t=t, context=context)

    log.info("Stage : Instantiate optimiser")
    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
    opt_state = optimiser.init(params)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        train_state = restore(ckpt_path)
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_state=opt_state,
            model_state=state,
            step=0,
            params=params,
            ema_rate=cfg.ema_rate,
            params_ema=params,
            rng=next_rng,  # TODO: we should actually use this for reproducibility
        )
        save(ckpt_path, train_state)

    if cfg.mode == "train" or cfg.mode == "all":
        # if train_state.step == 0 and cfg.test_test:
        #     evaluate(train_state, "test", step=cfg.steps)
        if train_state.step == 0 and cfg.test_plot:
            generate_plots(train_state, "test", step=-1)
        log.info("Stage : Training")
        train_state, success = train(train_state)
    if cfg.mode == "test" or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        if cfg.test_val:
            evaluate(train_state, "val", step=cfg.steps)
        if cfg.test_test:
            evaluate(train_state, "test", step=cfg.steps)
        if cfg.test_plot:
            generate_plots(train_state, "test", step=cfg.steps)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")
