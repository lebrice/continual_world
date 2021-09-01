from typing import Tuple
from continual_world.sequoia_integration.base_sac_method import GradientsTuple
import tensorflow as tf


class AgemHelper:
    def adjust_gradients(
        self, new_grads: GradientsTuple, ref_grads: GradientsTuple
    ) -> Tuple[GradientsTuple, int]:
        new_actor_grads, new_critic_grads, alpha_grad = new_grads
        ref_actor_grads, ref_critic_grads, _ = ref_grads

        dot_prod = 0.0
        ref_squared_norm = 0.0
        for new_grad, ref_grad in zip(new_actor_grads, ref_actor_grads):
            dot_prod += tf.reduce_sum(new_grad * ref_grad)
            ref_squared_norm += tf.reduce_sum(ref_grad * ref_grad)

        for new_grad, ref_grad in zip(new_critic_grads, ref_critic_grads):
            dot_prod += tf.reduce_sum(new_grad * ref_grad)
            ref_squared_norm += tf.reduce_sum(ref_grad * ref_grad)

        violation = tf.cond(dot_prod >= 0, lambda: 0, lambda: 1)

        actor_proj_grads = self.project_gradients(
            new_actor_grads, ref_actor_grads, dot_prod, ref_squared_norm
        )
        critic_proj_grads = self.project_gradients(
            new_critic_grads, ref_critic_grads, dot_prod, ref_squared_norm
        )

        gradients = GradientsTuple(
            actor_gradients=actor_proj_grads, critic_gradients=critic_proj_grads, alpha_gradient=alpha_grad
        )

        return gradients, violation

    def project_gradients(self, new_grads, ref_grads, dot_prod, ref_squared_norm):
        projected_grads = []
        for new_grad, ref_grad in zip(new_grads, ref_grads):
            projected_grads += [
                tf.cond(
                    dot_prod >= 0,
                    lambda: new_grad,
                    lambda: new_grad - (dot_prod / ref_squared_norm * ref_grad),
                )
            ]
        return projected_grads

