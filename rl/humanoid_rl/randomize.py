"""Per-environment model randomization for brax training (`randomization_fn`).

Each parallel environment gets its own copy of the randomized fields; the in_axes tree tells
brax which fields are batched. Ranges come from `config.domain_randomization_config()`.
"""

from typing import Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from humanoid_rl import config as cfg
from humanoid_rl import constants as c

FIELDS = ('geom_friction', 'body_mass', 'body_inertia', 'body_ipos', 'actuator_gainprm',
          'actuator_biasprm', 'dof_damping', 'dof_frictionloss', 'dof_armature')


def make_randomization_fn(mj_model: mujoco.MjModel,
                          dr: Optional[config_dict.ConfigDict] = None):
    """Return `fn(mjx_model, rng) -> (model, in_axes)` or None when randomization is disabled."""
    dr = dr if dr is not None else cfg.domain_randomization_config()
    if not dr.enabled:
        return None
    contact_geoms = jp.array([mj_model.geom(name).id for name in (c.FLOOR_GEOM, *c.FOOT_GEOMS)])
    torso = mj_model.body(c.TORSO_BODY).id
    joint_dofs = jp.arange(6, mj_model.nv)

    def uniform(rng, bounds, shape=()):
        return jax.random.uniform(rng, shape, minval=bounds[0], maxval=bounds[1])

    def single(model: mjx.Model, rng: jax.Array) -> dict:
        keys = jax.random.split(rng, 9)
        # Contact friction is the max over the two geoms, so floor and feet share one value.
        friction = model.geom_friction.at[contact_geoms, 0].set(uniform(keys[0], dr.friction))

        mass_scale = uniform(keys[1], dr.body_mass_scale, (model.nbody,))
        mass = model.body_mass * mass_scale
        inertia = model.body_inertia * mass_scale[:, None]
        torso_mass = mass[torso] + uniform(keys[2], dr.torso_mass_add)
        inertia = inertia.at[torso].multiply(torso_mass / mass[torso])
        mass = mass.at[torso].set(torso_mass)
        ipos = model.body_ipos.at[torso].add(jax.random.uniform(
            keys[3], (3,), minval=-dr.torso_com_offset, maxval=dr.torso_com_offset))

        kp = model.actuator_gainprm[:, 0] * uniform(keys[4], dr.kp_scale, (model.nu,))
        kd = -model.actuator_biasprm[:, 2] * uniform(keys[5], dr.kd_scale, (model.nu,))
        gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        biasprm = model.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)

        n = joint_dofs.shape[0]
        damping = model.dof_damping.at[joint_dofs].multiply(
            uniform(keys[6], dr.damping_scale, (n,)))
        frictionloss = model.dof_frictionloss.at[joint_dofs].multiply(
            uniform(keys[7], dr.frictionloss_scale, (n,)))
        armature = model.dof_armature.at[joint_dofs].multiply(
            uniform(keys[8], dr.armature_scale, (n,)))
        return dict(geom_friction=friction, body_mass=mass, body_inertia=inertia, body_ipos=ipos,
                    actuator_gainprm=gainprm, actuator_biasprm=biasprm, dof_damping=damping,
                    dof_frictionloss=frictionloss, dof_armature=armature)

    def randomize(model: mjx.Model, rng: jax.Array):
        values = jax.vmap(single, in_axes=(None, 0))(model, rng)
        model_v = model.tree_replace(values)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({field: 0 for field in FIELDS})
        return model_v, in_axes

    return randomize
