# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Plugin-owned adapters reconstructed exclusively from tensor/numeric weights.

These namespaces belong to the prepared payload, not SGLang. They allow reuse of
its existing K3 providers without receiving a model or runner from the host.
"""

from types import SimpleNamespace as NS


def norm(weight, eps):
    return NS(weight=weight, variance_epsilon=eps, eps=eps)


def linear(weight):
    return NS(weight=weight)


class WeightModel:
    def __init__(self, weights):
        if weights.abi_version != 1 or len(weights.layers) != 93:
            raise ValueError("K31 verify weight ABI requires all 93 layers")
        self.weights = weights
        self.config = NS(
            full_attention_layer_ids=weights.full_attention_ids,
            linear_layer_ids=weights.linear_attention_ids,
            activation_situ_beta=weights.situ_beta,
            activation_situ_linear_beta=weights.situ_linear_beta,
            rms_norm_eps=weights.rms_eps,
            moe_renormalize=bool(weights.moe_renormalize),
            num_hidden_layers=93,
            attn_res_block_size=weights.attn_res_block_size,
        )
        layers = []
        self.packed = []
        for record in weights.layers:
            a, sc = record.attention, record.attention_scalars
            self.packed.append(a[0])
            if record.attention_kind == 0:
                attn = NS(
                    q_a_layernorm=norm(a[1], sc[0]),
                    kv_a_layernorm=norm(a[2], sc[1]),
                    q_b_proj=linear(a[3]),
                    w_kc=a[4],
                    w_vc=a[5],
                    o_proj=linear(a[6]),
                    scaling=sc[2],
                    w_scale=sc[3],
                    attn_mqa=NS(q_scale_float=sc[4], k_scale_float=sc[5]),
                    indexer=NS(
                        skip_rope=True,
                        apply_hadamard=False,
                        wk=linear(a[7]),
                        k_norm=NS(weight=a[8], bias=a[9], variance_epsilon=sc[6]),
                    ),
                )
            else:
                attn = NS(
                    qkv_conv1d=NS(weight=a[1], bias=None),
                    f_b_proj=linear(a[2]),
                    A_log=a[3],
                    dt_bias=a[4],
                    o_norm=norm(a[5], sc[1]),
                    o_proj=linear(a[6]),
                    attn=NS(lower_bound=sc[0]),
                )
            f = record.feed_forward
            fs = record.feed_forward_scalars
            if record.is_moe:
                mlp = NS(
                    experts=NS(
                        w13_weight=f[0],
                        w13_weight_scale=f[1],
                        w2_weight=f[2],
                        w2_weight_scale=f[3],
                    ),
                    shared_experts=NS(
                        gate_up_proj=linear(f[4]), down_proj=linear(f[5])
                    ),
                    gate=NS(weight=f[6], e_score_correction_bias=f[7]),
                    routed_expert_down_proj=linear(f[8]),
                    routed_expert_up_proj=linear(f[9]),
                    routed_expert_norm=norm(f[10], weights.rms_eps),
                    routed_scaling_factor=fs[0],
                )
            else:
                mlp = NS(
                    gate_up_proj=linear(f[0]),
                    down_proj=linear(f[1]),
                    act_fn=NS(
                        beta=weights.situ_beta, linear_beta=weights.situ_linear_beta
                    ),
                )
            t, e = record.residual, record.residual_eps
            layer = NS(
                layer_idx=record.layer_id,
                _is_moe_layer=bool(record.is_moe),
                self_attn=attn,
                mlp=mlp,
                self_attention_res_norm=norm(t[0], e[0]),
                self_attention_res_proj=linear(t[1]),
                input_layernorm=norm(t[2], e[1]),
                mlp_res_norm=norm(t[3], e[2]),
                mlp_res_proj=linear(t[4]),
                post_attention_layernorm=norm(t[5], e[3]),
                prev_valid_blocks=record.previous_blocks,
                is_block_write_layer=bool(record.writes_block),
                attn_res_block_size=weights.attn_res_block_size,
            )
            layers.append(layer)
        w, e = weights.final_weights, weights.final_eps
        self.model = NS(
            layers=tuple(layers),
            start_layer=0,
            end_layer=93,
            config=self.config,
            embed_tokens=linear(weights.embedding),
            output_attn_res_norm=norm(w[0], e[0]),
            output_attn_res_proj=linear(w[1]),
            norm=norm(w[2], e[1]),
        )

    def parameters(self):
        yield self.weights.embedding
        yield from self.weights.final_weights
        for layer in self.weights.layers:
            yield from layer.attention
            yield from layer.residual
            yield from layer.feed_forward

    def buffers(self):
        return iter(())

    def modules(self):
        # Inventory excludes all tensor storages above; there is no host model tree.
        yield self
        yield self.model
