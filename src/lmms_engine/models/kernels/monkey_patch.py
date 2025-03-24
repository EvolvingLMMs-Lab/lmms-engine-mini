import inspect
from functools import partial
from typing import Callable

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print(
        "liger kernel not installed, please install it with `pip install liger-kernel`"
    )

from transformers import PreTrainedModel

from ...utils.logging_utils import Logging


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(
    module, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True
):
    module.offset = offset
    module.casting_mode = casting_mode
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    )
    module.in_place = in_place
    _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)


def apply_liger_kernel_to_kino_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2
    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    from ..qwen2_5_vl_audio import modeling_qwen2_5_vl as kino_modeling_qwen2_5_vl
    from .qwen2_5_vl_liger import lce_forward as qwen2_5_vl_lce_forward

    if use_rmpad:
        qwen2_5_vl_lce_forward = partial(qwen2_5_vl_lce_forward, use_rmpad=use_rmpad)

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
            liger_multimodal_rotary_pos_emb
        )
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        kino_modeling_qwen2_5_vl.KinoQwen2_5_VLForConditionalGeneration.forward = (
            qwen2_5_vl_lce_forward
        )
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .rmpad.qwen2_ops import attn_forward as qwen2_ops_attn_forward
        from .rmpad.qwen2_ops import (
            decoder_layer_forward as qwen2_ops_decoder_layer_forward,
        )
        from .rmpad.qwen2_ops import model_forward as qwen2_ops_model_forward

        modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_ops_model_forward
        modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = (
            qwen2_ops_decoder_layer_forward
        )
        modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen2_ops_attn_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2_5_VLModel = getattr(model, model.base_model_prefix, model)

        if hasattr(model, "visual"):
            # Patch Qwen2_5_VisionTransformerPretrainedModel
            for vision_block in model.visual.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "kino_qwen2_5_vl": apply_liger_kernel_to_kino_qwen2_5_vl,
}


def _apply_liger_kernel(model_type: str, **kwargs) -> None:
    """
    Applies Liger kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    ** Note: Calling _apply_liger_kernel() after model initialization
    will not be able to fully patch models. This must be called before model initialization.
    If the model has already been instantiated

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    if not model_type:
        Logging.info("Model type was not provided. No Liger kernels will be applied.")
        return

    if model_type not in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        Logging.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }

    Logging.info(
        f"Applying Liger kernels for model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)


def _apply_liger_kernel_to_instance(model: PreTrainedModel, **kwargs) -> None:
    """
    Applies Liger kernels to the provided model instance.

    Args:
        - model: the model instance to apply Liger kernels to
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    model_type = getattr(model, "config", None) and getattr(
        model.config, "model_type", None
    )

    if not model_type:
        Logging.info(
            "Model type could not be determined from model config. No Liger kernels will be applied."
        )
        return

    if model_type not in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        Logging.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]

    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }
    Logging.info(
        f"Applying Liger kernels to model instance with model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    apply_fn(model=model, **applicable_kwargs)
