import torch
import math
import os
import ldm_patched.modules.utils
import ldm_patched.modules.model_management
import ldm_patched.modules.model_detection
import ldm_patched.modules.model_patcher
import ldm_patched.modules.ops

import ldm_patched.controlnet.cldm
import ldm_patched.t2ia.adapter
import ldm_patched.modules.latent_formats
import ldm_patched.ldm.lumina.controlnet


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.global_average_pooling = False
        self.timestep_range = None

        if device is None:
            device = ldm_patched.modules.model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0)):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1]))
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        if control_input is not None:
            for i in range(len(control_input)):
                key = 'input'
                x = control_input[i]
                if x is not None:
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].insert(0, x)

        if control_output is not None:
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = 'middle'
                    index = 0
                else:
                    key = 'output'
                    index = i
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)
        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            if o[i].shape[0] < prev_val.shape[0]:
                                o[i] = prev_val + o[i]
                            else:
                                o[i] += prev_val
        return out

class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(device)
        self.control_model = control_model
        self.load_device = load_device
        self.control_model_wrapped = ldm_patched.modules.model_patcher.ModelPatcher(self.control_model, load_device=load_device, offload_device=ldm_patched.modules.model_management.unet_offload_device())
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = ldm_patched.modules.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond['c_crossattn']
        y = cond.get('y', None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(dtype), y=y)
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()

class ControlLoraOps:
    class Linear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            weight, bias = ldm_patched.modules.ops.cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.linear(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias)
            else:
                return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None


        def forward(self, input):
            weight, bias = ldm_patched.modules.ops.cast_bias_weight(self, input)
            if self.up is not None:
                return torch.nn.functional.conv2d(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                return torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, device=None):
        ControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:
            class control_lora_ops(ControlLoraOps, ldm_patched.modules.ops.disable_weight_init):
                pass
        else:
            class control_lora_ops(ControlLoraOps, ldm_patched.modules.ops.manual_cast):
                pass
            dtype = self.manual_cast_dtype

        controlnet_config["operations"] = control_lora_ops
        controlnet_config["dtype"] = dtype
        self.control_model = ldm_patched.controlnet.cldm.ControlNet(**controlnet_config)
        self.control_model.to(ldm_patched.modules.model_management.get_torch_device())
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()

        for k in sd:
            weight = sd[k]
            try:
                ldm_patched.modules.utils.set_attr(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                ldm_patched.modules.utils.set_attr(self.control_model, k, self.control_weights[k].to(dtype).to(ldm_patched.modules.model_management.get_torch_device()))

    def copy(self):
        c = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return ldm_patched.modules.utils.calculate_parameters(self.control_weights) * ldm_patched.modules.model_management.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)

def load_controlnet(ckpt_path, model=None):
    controlnet_data = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    if "lora_controlnet" in controlnet_data:
        return ControlLora(controlnet_data)

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        unet_dtype = ldm_patched.modules.model_management.unet_dtype()
        controlnet_config = ldm_patched.modules.model_detection.unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
        if controlnet_config is None:
            raise RuntimeError(f"ERROR: Could not detect controlnet model type of: {ckpt_path}")
        diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            print("leftover keys:", leftover_keys)
        controlnet_data = new_sd

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        unet_dtype = ldm_patched.modules.model_management.unet_dtype()
        model_config = ldm_patched.modules.model_detection.model_config_from_unet(controlnet_data, prefix, unet_dtype, use_base_if_no_match=True)
        if model_config is None:
            raise RuntimeError(f"ERROR: Could not detect controlnet model type of: {ckpt_path}")
        controlnet_config = model_config.unet_config
    load_device = ldm_patched.modules.model_management.get_torch_device()
    manual_cast_dtype = ldm_patched.modules.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = ldm_patched.modules.ops.manual_cast
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = ldm_patched.controlnet.cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                ldm_patched.modules.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    print(missing, unexpected)

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control

class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(x_noisy.shape[3] * 8, x_noisy.shape[2] * 8)
            self.cond_hint = ldm_patched.modules.utils.common_upscale(self.cond_hint_original, width, height, 'nearest-exact', "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = list(map(lambda a: None if a is None else a.clone(), self.control_input))
        mid = None
        if self.t2i_model.xl == True:
            mid = control_input[-1:]
            control_input = control_input[:-1]
        return self.control_merge(control_input, mid, control_prev, x_noisy.dtype)

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in)
        self.copy_to(c)
        return c

def load_t2i_adapter(t2i_data):
    if 'adapter' in t2i_data:
        t2i_data = t2i_data['adapter']
    if 'adapter.body.0.resnets.0.block1.weight' in t2i_data: #diffusers format
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace["adapter.body.{}.resnets.{}.".format(i, j)] = "body.{}.".format(i * 2 + j)
            prefix_replace["adapter.body.{}.".format(i, j)] = "body.{}.".format(i * 2)
        prefix_replace["adapter."] = ""
        t2i_data = ldm_patched.modules.utils.state_dict_prefix_replace(t2i_data, prefix_replace)
    keys = t2i_data.keys()

    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = ldm_patched.t2ia.adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        xl = False
        if cin == 256 or cin == 768:
            xl = True
        model_ad = ldm_patched.t2ia.adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv, xl=xl)
    else:
        return None
    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        print("t2i missing", missing)

    if len(unexpected) > 0:
        print("t2i unexpected", unexpected)

    return T2IAdapter(model_ad, model_ad.input_channels)


def _z_image_controlnet_convert(sd):
    """Rename HF/diffusers-style attention keys (separate to_q/to_k/to_v
    projections, norm_q/norm_k, to_out.0) in a Z-Image-Turbo-Fun-Controlnet
    checkpoint to this vendored NextDiT's fused qkv/out/q_norm/k_norm
    naming (ldm_patched.ldm.lumina.model.JointAttention). Ported from
    ComfyUI's comfy_extras/nodes_model_patch.py:z_image_convert -- the
    to_q/to_k/to_v concatenation order (Q, K, V) matches JointAttention.qkv's
    output layout exactly because dict iteration is alphabetical
    ("to_k" < "to_q" < "to_v").
    """
    replace_keys = {
        ".attention.to_out.0.bias": ".attention.out.bias",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.to_out.0.weight": ".attention.out.weight",
    }

    out_sd = {}
    cc = []
    for k in sorted(sd.keys()):
        w = sd[k]
        k_out = k
        if k_out.endswith(".attention.to_k.weight"):
            cc = [w]
            continue
        if k_out.endswith(".attention.to_q.weight"):
            cc = [w] + cc
            continue
        if k_out.endswith(".attention.to_v.weight"):
            cc = cc + [w]
            w = torch.cat(cc, dim=0)
            k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")

        for r, rr in replace_keys.items():
            k_out = k_out.replace(r, rr)
        out_sd[k_out] = w

    return out_sd


def _detect_zimage_controlnet_config(sd):
    """Infer ZImage_Control's `n_control_layers`/`additional_in_dim`/
    `refiner_control` constructor kwargs from a (already qkv-converted)
    checkpoint's own key layout -- verified against the real published
    safetensors headers of both the "lite" (3-control-layer) and "-2.1"
    Union (15-control-layer) checkpoints, not guessed. Ported from
    ComfyUI's comfy_extras/nodes_model_patch.py:ModelPatchLoader's Z-Image
    branch. Returns {} (ZImage_Control's own defaults) for an unrecognized
    layout -- load_controlnet_zimage's strict state-dict load is what
    actually rejects an incompatible checkpoint, not this function.
    """
    config = {}
    if 'control_layers.4.adaLN_modulation.0.weight' not in sd:
        config['n_control_layers'] = 3
        config['additional_in_dim'] = 17
        config['refiner_control'] = True
    if 'control_layers.14.adaLN_modulation.0.weight' in sd:
        config['n_control_layers'] = 15
        config['additional_in_dim'] = 17
        config['refiner_control'] = True
    return config


def load_controlnet_zimage(ckpt_path):
    """Load an Alibaba/Tongyi-MAI Z-Image-Turbo-Fun-Controlnet checkpoint
    (e.g. `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`) into a
    ldm_patched.ldm.lumina.controlnet.ZImage_Control and return it, ready to
    be wrapped by ZImageControlNetPatch and applied via
    ModelPatcher.set_model_double_block_patch/set_model_noise_refiner_patch.

    Only control-only extractions of this checkpoint family are supported
    (no base-model weights bundled in); a checkpoint that does not match
    the expected key layout raises RuntimeError rather than silently
    loading a partial/incorrect model.
    """
    sd = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = _z_image_controlnet_convert(sd)
    config = _detect_zimage_controlnet_config(sd)

    load_device = ldm_patched.modules.model_management.get_torch_device()
    dtype = ldm_patched.modules.model_management.unet_dtype()
    manual_cast_dtype = ldm_patched.modules.model_management.unet_manual_cast(dtype, load_device)
    if manual_cast_dtype is not None:
        operations = ldm_patched.modules.ops.manual_cast
        dtype = manual_cast_dtype
    else:
        operations = ldm_patched.modules.ops.disable_weight_init

    control_model = ldm_patched.ldm.lumina.controlnet.ZImage_Control(
        dtype=dtype, device=load_device, operations=operations, **config
    )
    missing, unexpected = control_model.load_state_dict(sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            "Z-Image ControlNet checkpoint at '{}' does not match the expected "
            "control-only key layout ({} missing key(s), {} unexpected key(s)). "
            "Only control-only extractions of the Z-Image-Turbo-Fun-Controlnet-Union "
            "family (e.g. the '-2.1' Union checkpoint) are supported.".format(
                ckpt_path, len(missing), len(unexpected)
            )
        )
    control_model.eval()
    return control_model


def _encode_zimage_control_hint(vae, image, additional_in_dim):
    """VAE-encode a pixel-space hint image into the Flux-format, Fun-style
    latent layout the Z-Image-Turbo-Fun-Controlnet checkpoint was trained
    on: `[control_latent(16ch), mask(1ch), inpaint_latent(16ch)]` when the
    checkpoint has an inpaint/mask branch (`additional_in_dim > 0`, true for
    the "-2.1" Union checkpoint this ticket targets), or just the plain
    control latent otherwise. No inpaint image/mask is supported yet (Canny
    text-to-image only per this ticket's scope), so the inpaint branch is
    filled with a neutral all-gray image and an all-zero ("nothing masked")
    mask -- matching ComfyUI's own no-inpaint-input fallback
    (comfy_extras/nodes_model_patch.py:ZImageControlPatch.encode_latent_cond).
    """
    latent_format = ldm_patched.modules.latent_formats.Flux()
    latent_image = latent_format.process_in(vae.encode(image[:, :, :, :3]))
    if additional_in_dim <= 0:
        return latent_image

    blank = torch.ones_like(image) * 0.5
    inpaint_latent = latent_format.process_in(vae.encode(blank[:, :, :, :3]))
    mask = torch.zeros_like(inpaint_latent)[:, :1]
    return torch.cat([latent_image, mask, inpaint_latent], dim=1)


class ZImageControlNetPatch:
    """Drives a ldm_patched.ldm.lumina.controlnet.ZImage_Control model one
    main NextDiT block at a time, from NextDiT.forward()'s "double_block"/
    "noise_refiner" hooks (ldm_patched/ldm/lumina/model.py, wired via
    ModelPatcher.set_model_double_block_patch/set_model_noise_refiner_patch).
    Ported from ComfyUI's comfy_extras/nodes_model_patch.py:ZImageControlPatch.

    The main model's forward loop calls this once per noise_refiner/main
    layer block; this class lazily advances the control stack (far shorter
    than the main model's 30 layers) in lockstep and adds its
    zero-initialized residual ("hint") onto the main model's running image
    tokens at the matching block index, spaced evenly across the main
    model's block count (`div = round(total_blocks / n_control_layers)`).

    Unlike the SDXL/UNet ControlNet path (`ControlNet.get_control()`,
    conditioning-based), this class patches the unet ModelPatcher directly
    -- the same "clone, then set_model_*_patch" shape as
    `modules/inpaint_worker.py:InpaintWorker.patch()` and
    `extras/ip_adapter.py:patch_model()` in this codebase.
    """

    # Sentinel `_state[0]` values used the first time each phase touches a
    # freshly (re-)embedded control context, before any real block has run --
    # distinct so a mid-noise_refiner-phase state is never mistaken for a
    # mid-double_block-phase one (see the reset condition below).
    _NOISE_REFINER_START = -3
    _DOUBLE_BLOCK_START = -1

    def __init__(self, control_model, vae, image, strength):
        self.control_model = control_model
        self.strength = strength
        self._encoded_hint = _encode_zimage_control_hint(vae, image, control_model.additional_in_dim)
        self._state = None

    def __call__(self, kwargs):
        img = kwargs["img"]
        img_input = kwargs["img_input"]
        pe = kwargs["pe"]
        vec = kwargs["vec"]
        block_index = kwargs["block_index"]
        block_type = kwargs.get("block_type", "double")

        n_control_layers = self.control_model.n_control_layers
        total_blocks = kwargs.get("transformer_options", {}).get("total_blocks")
        if not total_blocks:
            return kwargs
        div = max(1, round(total_blocks / n_control_layers))
        cnet_index = block_index // div
        cnet_index_float = block_index / div

        # For the double_block (main layer) phase this guards trailing main
        # blocks past the last control layer's mapped position (e.g. when
        # `n_control_layers` does not evenly divide `total_blocks`), which
        # otherwise would never hit the exact-index match below and would
        # leave `_state` stuck non-None into the next forward pass. During
        # the noise_refiner phase `block_index` is always tiny (0/1) so this
        # is a no-op there; kept unconditional to match the reference.
        if cnet_index_float > (n_control_layers - 1):
            self._state = None
            return kwargs

        # A fresh embed is needed at the start of every forward pass (a new
        # denoising step restarts NextDiT.forward from noise_refiner block 0)
        # and, defensively, if `_state` ever overshoots the block this call
        # is asking for -- e.g. the noise_refiner -> double_block phase
        # transition within one forward pass, where `_state[0]` still holds
        # the noise_refiner phase's own (unrelated) counter value. Mirrors
        # ComfyUI's `temp_data is None or temp_data[0] > cnet_index` guard.
        if self._state is None or self._state[0] > cnet_index:
            start = self._NOISE_REFINER_START if block_type == "noise_refiner" else self._DOUBLE_BLOCK_START
            hint = self._encoded_hint.to(device=img.device, dtype=img.dtype)
            embedded = self.control_model.embed_hint(hint, freqs_cis=pe, adaln_input=vec)
            self._state = (start, (None, embedded))

        if block_type == "noise_refiner":
            # control_noise_refiner has exactly as many blocks as the main
            # model's own noise_refiner (always 2) -- a direct 1:1 mapping,
            # no div/ratio needed, so block_index is used as the layer id.
            control_context = self._state[1][1]
            next_state = self._state[0] + 1
            hint_out, control_context = self.control_model.forward_noise_refiner_block(
                block_index, control_context, img_input[:, :control_context.shape[1]], pe, vec,
            )
            self._state = (next_state, (hint_out, control_context))
            if hint_out is not None:
                n = hint_out.shape[1]
                img = img.clone()
                img[:, :n] = img[:, :n] + hint_out * self.strength
                kwargs["img"] = img
            return kwargs

        while self._state[0] < cnet_index and (self._state[0] + 1) < n_control_layers:
            control_context = self._state[1][1]
            next_layer = self._state[0] + 1
            hint_out, control_context = self.control_model.forward_control_block(
                next_layer, control_context, img_input[:, :control_context.shape[1]], pe, vec,
            )
            self._state = (next_layer, (hint_out, control_context))

        if cnet_index_float == self._state[0]:
            hint_out = self._state[1][0]
            n = hint_out.shape[1]
            img = img.clone()
            img[:, :n] = img[:, :n] + hint_out * self.strength
            kwargs["img"] = img
            if n_control_layers == self._state[0] + 1:
                self._state = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self._encoded_hint = self._encoded_hint.to(device_or_dtype)
            self._state = None
        return self
