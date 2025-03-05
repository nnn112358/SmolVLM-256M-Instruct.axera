import torch
from torch import nn
import onnxruntime as ort
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3VisionTransformer,
    Idefics3Connector,
    Idefics3BaseModelOutputWithPast,
    IDEFICS3_INPUTS_DOCSTRING,
)
from transformers.models.idefics3.configuration_idefics3 import (
    Idefics3Config,
    Idefics3VisionConfig,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import add_start_docstrings_to_model_forward


class Idefics3VisionTransformerExport(Idefics3VisionTransformer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(
                dtype=torch.bool, device=pixel_values.device
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.connector(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_onnx(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        print("test onnx--------------")

        device = pixel_values.device

        session = ort.InferenceSession(
            "SmolVLM-256M-Instruct_vision.onnx", providers=["CPUExecutionProvider"]
        )

        outputs = []
        for i in range(pixel_values.shape[0]):
            inputs = {
                "pixel_values": pixel_values[i : i + 1]
                .cpu()
                .numpy()
                .astype(np.float32),
            }
            out = session.run(["last_hidden_state"], inputs)[0]
            outputs.append(out)
        outputs = np.concatenate(outputs, axis=0)

        last_hidden_state = torch.from_numpy(outputs).to(device)

        if not return_dict:
            return (last_hidden_state,) + (None, None)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=None, attentions=None
        )

    def forward_export(
        self,
        pixel_values,
    ) -> Union[Tuple, BaseModelOutput]:
        patch_attention_mask = None
        output_attentions = False
        output_hidden_states = False

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(
                dtype=torch.bool, device=pixel_values.device
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.connector(last_hidden_state)

        return last_hidden_state


class Idefics3ModelExport(Idefics3Model):
    def __init__(self, config: Idefics3Config):
        super().__init__(config)

        self.vision_model = Idefics3VisionTransformerExport._from_config(
            config.vision_config
        )

        # move connector into vision_model
        self.vision_model.connector = self.connector

    @add_start_docstrings_to_model_forward(
        """
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        IDEFICS3_INPUTS_DOCSTRING,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Idefics3BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError(
                "When first calling the model, if input_embeds are passed, input_ids should not be None."
            )

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(
                self.device
            )

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "You cannot specify both pixel_values and image_hidden_states at the same time"
            )
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(
                batch_size * num_images, *pixel_values.shape[2:]
            )

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(
                dim=(-1, -2, -3)
            ) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(
                        pixel_values.size(0),
                        pixel_values.size(2),
                        pixel_values.size(3),
                    ),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[
                    real_images_inds
                ].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(
                dimension=1, size=patch_size, step=patch_size
            )
            patches_subgrid = patches_subgrid.unfold(
                dimension=2, size=patch_size, step=patch_size
            )
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            # image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.dtype, device=input_ids.device
            )

        if (
            past_seen_tokens == 0
            and inputs_embeds is not None
            and image_hidden_states is not None
        ):
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class Idefics3ForConditionalGenerationExport(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Idefics3ModelExport(config)
