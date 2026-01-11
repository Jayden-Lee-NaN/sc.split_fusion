import torch
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F


from splitfuser.modules.base_module import BaseModule
from splitfuser.utils import SplitConfig


class SplitAttentionPP(BaseModule):
    def __init__(self, module: Attention, split_config: SplitConfig):
        super(SplitAttentionPP, self).__init__(module, split_config)

        to_k = module.to_k
        to_v = module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = nn.Linear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv


class SplitCrossAttentionPP(SplitAttentionPP):
    def __init__(self, module: Attention, split_config: SplitConfig):
        super(SplitCrossAttentionPP, self).__init__(module, split_config)
        self.kv_cache = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        assert encoder_hidden_states is not None
        recompute_kv = self.counter == 0

        attn = self.module
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if recompute_kv or self.kv_cache is None:
            kv = self.to_kv(encoder_hidden_states)
            self.kv_cache = kv
        else:
            kv = self.kv_cache
        key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.counter += 1

        return hidden_states


class SplitSelfAttentionPP(SplitAttentionPP):
    def __init__(self, module: Attention, split_config: SplitConfig):
        super(SplitSelfAttentionPP, self).__init__(module, split_config)
        self.last_full_kv = None

        self.split_list = []
        self.send_shapes = []

        self.other_send_shapes: list[list[int]] = []
        self.slice_h: list[list[int]] = []
        self.other_send_times: list[int] = []

        self.send_times = 1
        self.kv_bucket_shape = []

    def _get_send_shapes(self, idx: int, split_list: list[int], bucket_size: int) -> list[int]:
        send_shapes = []
        if split_list[idx] < bucket_size:
            send_shapes.append(split_list[idx])
        else:
            for i in range(1, (split_list[idx] // bucket_size) + 1):
                send_shapes.append(bucket_size)
            if split_list[idx] % bucket_size != 0:
                send_shapes.append(split_list[idx] % bucket_size)
        send_shapes = [sum(send_shapes[:i]) for i in range(len(send_shapes) + 1)]
        return send_shapes

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module
        split_config = self.split_config
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        encoder_hidden_states = hidden_states

        kv = self.to_kv(encoder_hidden_states)

        if split_config.n_device_per_batch == 1:
            full_kv = kv
        else:
            if self.buffer_list is None:
                full_kv = torch.cat([kv for _ in range(split_config.n_device_per_batch)], dim=1)
                self.kv_bucket_shape = [full_kv.shape[0], full_kv.shape[1] // (self.split_config.n_device_per_batch), full_kv.shape[2]]

            elif split_config.mode == "full_sync" or self.counter <= split_config.warmup_steps:
                dist.all_gather(self.buffer_list, kv, group=split_config.batch_group, async_op=False)
                full_kv = torch.cat(self.buffer_list, dim=1)
            else:

                if self.counter == split_config.warmup_steps + 1:
                    self.split_list = self.split_config.get_split_list(self.last_full_kv.shape[1])
                    self.slice_h = [sum(self.split_list[:i]) for i in range(len(self.split_list) + 1)]
                    self.send_shapes = self._get_send_shapes(split_config.split_idx(), self.split_list, self.kv_bucket_shape[1])

                    for i in range(split_config.n_device_per_batch):
                        self.other_send_times.append(1)
                        self.other_send_shapes.append(self._get_send_shapes(i, self.split_list, self.kv_bucket_shape[1]))

                kv_bucket = torch.empty(self.kv_bucket_shape, device=kv.device)

                if self.send_times + 1 == len(self.send_shapes):
                    kv_bucket[:, : self.send_shapes[-1] - self.send_shapes[-2], :] = kv[:, self.send_shapes[-2] : self.send_shapes[-1], :]
                    self.send_times = 1
                else:
                    kv_bucket = kv[:, self.send_shapes[self.send_times - 1] : self.send_shapes[self.send_times], :]
                    self.send_times += 1

                full_kv = self.last_full_kv

                for i, buffer in enumerate(self.buffer_list):
                    if self.other_send_times[i] + 1 == len(self.other_send_shapes[i]):
                        data = buffer[:, : self.other_send_shapes[i][-1] - self.other_send_shapes[i][-2], :]
                        full_kv[:, self.slice_h[i] + self.other_send_shapes[i][-2] : self.slice_h[i] + self.other_send_shapes[i][-1], :] = data
                        self.other_send_times[i] = 1
                    else:
                        full_kv[
                            :, self.slice_h[i] + self.other_send_shapes[i][self.other_send_times[i] - 1] : self.slice_h[i] + self.other_send_shapes[i][self.other_send_times[i]], :
                        ] = buffer
                        self.other_send_times[i] += 1

                full_kv[:, self.slice_h[split_config.split_idx()] : self.slice_h[split_config.split_idx() + 1], :] = kv

                if split_config.mode != "no_sync":
                    self.comm_manager.enqueue(self.idx, kv_bucket)

        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.last_full_kv = full_kv
        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        split_config = self.split_config
        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                print(f"{self.split_config.rank}, {self.split_config.batch_idx()}: 等待数据")
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None
                print(f"{self.split_config.rank}, {self.split_config.batch_idx()}: 数据处理完毕")

        b, l, c = hidden_states.shape
        if split_config.n_device_per_batch > 1 and self.buffer_list is None:
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.register_tensor(shape=(b, l, self.to_kv.out_features), torch_dtype=hidden_states.dtype, layer_type="attn")
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
        output = self._forward(hidden_states, scale=scale)

        self.counter += 1
        return output
