# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.energon import VQASample, InterleavedSample

from nemo.collections.vlm.data.task_encoder import DataBatch, DataSample
from nemo.collections.vlm.data.task_encoder import TaskEncoder as BaseTaskEncoder
from nemo.collections.vlm.data.task_encoder import TaskEncoderConfig as BaseTaskEncoderConfig
from nemo.collections.vlm.data.utils import _find_pattern_indices
from nemo.utils import logging


@dataclass
class TaskEncoderConfig(BaseTaskEncoderConfig):
    """Configuration for Gemma 3 processing.

    This class consolidates all configuration needed for Gemma 3 processing,
    including model paths, tokenization, image processing, and sequence packing parameters.

    """

    stop_string: Optional[str] = ""
    system_prompt: Optional[str] = None
    image_token_str: str = "<start_of_image>"
    image_token_id: int = 262144 # This is the token id for <image_soft_token>


@dataclass
class Gemma3DataSample(DataSample):
    """Data Sample for Gemma3"""

    input_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class TaskEncoder(BaseTaskEncoder):
    """TaskEncoder for Gemma 3 data processing.

    This class handles the processing of different types of Gemma 3 samples,
    including Visual Question Answering (VQA), Captioning, and Interleaved samples.
    It provides functionality for encoding individual samples, batching them together,
    and handling packed sequences for efficient processing.

    The encoder supports:
    - VQA samples: Processing image-question pairs with corresponding answers
    - Interleaved samples: Processing alternating image and text content
    - [In progress] Similarity interleaved samples: Processing image-text pairs for similarity tasks
    - [In progress] Packed sequences: Efficient processing of multiple samples in a single sequence

    Args:
        config (TaskEncoderConfig): Configuration object containing processing parameters

    Note:
        When using packed sequences, the micro batch size must be 1, and the global batch
        size and sequence length must be adjusted accordingly.
    """

    def __init__(self, config: TaskEncoderConfig):
        """Initialize the Gemma 3 processor.

        Args:
            config (TaskEncoderConfig): Configuration for processing
        """
        self.config = config
        self.hf_processor = self.config.hf_processor
        self.tokenizer = self.config.tokenizer

        # Initialize encoders with the config
        self.encoders = {
            "VQASample": self.encode_vqa_sample,
            "InterleavedSample": self.encode_interleaved_sample,
        }

    def encode_batch(self, batch_data: DataBatch) -> dict:
        """Encode a batched set of samples for model input.

        This method transforms the raw batched data into a format ready for model input, including
        generating position IDs and other necessary fields.

        Parameters:
            batch_data (DataBatch): The raw batch of data to be encoded.

        Returns:
            dict: A dictionary containing the encoded batch data, ready for model input.
        """
        batch_data = super().encode_batch(batch_data)
        batch_data["media"] = batch_data["media"].reshape(-1, *batch_data["media"].shape[2:])
        return batch_data

    def encode_vqa_sample_multi_turns(self, input_sample: VQASample):
        """Encode a VQA sample multi turns into a DataSample format.

        Args:
            input_sample (VQASample): Input VQA sample containing image, context and answers

        Returns:
            Encoded tokens, labels and images.
        """
        images = input_sample.image if isinstance(input_sample.image, list) else [input_sample.image]

        contexts = json.loads(input_sample.context.decode('utf-8'))
        messages = []
        if self.config.system_prompt:
            messages.append({'role': 'system', 'content': self.config.system_prompt})
        for context in contexts:
            messages.append(context)

        # Apply chat template and process with HF processor
        # `add_generation_prompt=False` because we're providing the full ground truth sequence
        # We remove the <bos> token using removeprefix('<bos>') since we're finetuning.
        # The Processor will add this token before training and the model expects only one.
        converted_messages = self.hf_processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        ).removeprefix('<bos>')
        outputs = self.hf_processor(
            images=images,
            text=converted_messages,
            return_tensors="pt",
            images_kwargs={"do_rescale": False},
        )
        # Get tokens and images from processor output
        # Squeeze the batch dimension as we process one sample at a time
        tokens = outputs["input_ids"].squeeze(0)
        images = outputs.get("pixel_values")  # Use .get() for optional images

        # --- Label Generation ---
        # Same as: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/vlm/qwen2vl/data/task_encoder.py#L263-L270
        # Initialize labels with ignore placeholder
        labels = torch.full_like(tokens, self.config.ignore_place_holder)
        search_start_index = 0
        for context in contexts:
            if context['role'] != 'assistant':
                continue
            # Tokenize the answer, including the stop string if provided
            answer_with_stop = (
                context['content'][0]['text'].rstrip().lstrip() + "<end_of_turn>" + (self.config.stop_string or "")
            )
            answer_with_stop = answer_with_stop.rstrip().lstrip()
            answer_tokens = self.tokenizer.tokenizer(answer_with_stop, add_special_tokens=False)["input_ids"]
            answer_tokens_tensor = torch.tensor(answer_tokens, device=tokens.device)  # Ensure same device

            # sometimes the tokenizer can add additional space. See:
            # https://github.com/huggingface/transformers/issues/25073#issuecomment-1655271420
            if self.tokenizer.tokenizer.decode(answer_tokens[0]) == "":
                answer_tokens_tensor = answer_tokens_tensor[1:]

            # Find answer pattern in tokens
            answer_start, answer_end = _find_pattern_indices(tokens, answer_tokens_tensor, search_start_index)
            if answer_start >= 0:
                labels[answer_start:answer_end] = tokens[answer_start:answer_end]
                search_start_index = answer_end
            else:
                logging.warning(
                    "Unable to find answer segment in the tokenized conversation. "
                    "Skipping labeling for this and subsequent answers. Details: "
                    "\n- Processed Text: %s"
                    "\n- Tokens: %s"
                    "\n- Target Answer Tokens: %s"
                    "\n- Search Start Index: %d",
                    converted_messages,
                    tokens,
                    answer_tokens,
                    search_start_index,
                )
                break
        return tokens, labels, images

    def encode_vqa_sample(self, input_sample: VQASample) -> DataSample:
        """Encode a VQA sample into a DataSample format.

        Args:
            input_sample (VQASample): Input VQA sample containing image, context and answers

        Returns:
            DataSample: Encoded sample with processed image, tokens, labels and loss mask
        """
        tokens, labels, images = self.encode_vqa_sample_multi_turns(input_sample)

        # Prepare final tensors
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        seqlen = len(tokens)  # Original sequence length before padding
        position_ids = torch.arange(seqlen, dtype=torch.int64)

        # Pad tokens and labels to a multiple of `pad_to_multiple_of` if specified
        if self.config.pad_to_multiple_of:
            tokens, labels = self.pad_tokens_and_labels(tokens, labels, seqlen)

        # Compute loss mask
        loss_mask = self.compute_loss_mask(labels)

        # Convert images to bfloat16 and stack, or create an empty tensor if no images
        if images is not None and images.numel() > 0:
            # Ensure images tensor is on the same device as tokens/labels if needed
            images = images.to(device=tokens.device, dtype=torch.bfloat16)
            processed_image = images  # Already stacked by HF processor if multiple images/frames
        else:
            # Create an empty tensor with appropriate dimensions and dtype if no images
            processed_image = None

        sample = Gemma3DataSample(
            __key__=input_sample.__key__,
            __restore_key__=input_sample.__restore_key__,
            __subflavor__=input_sample.__subflavor__,
            __subflavors__=input_sample.__subflavors__,
            pixel_values=processed_image,
            input_ids=tokens,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
        )

        return sample

    def tokenize_interleaved_sample(self, input_sample: InterleavedSample) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Tokenize the input sequence and process images in an interleaved sample.

        This method processes a sequence that consists of text strings and image tensors.
        The text is tokenized, and the images are processed. The method returns a tensor
        of tokenized text and a concatenated tensor of processed images.

        Parameters:
        sample (InterleavedSample): The interleaved sample containing a sequence of text strings and image tensors.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor with tokenized text and image token IDs.
            - A concatenated tensor of processed images.
        """
        texts, images = [], []
        for item in input_sample.sequence:
            if type(item) == str:
                texts.append(item)
            elif type(item) == torch.Tensor:
                images.append(item)
                texts.append(self.config.image_token_str)  # Append start token to the last text. HF Processor will replace this token with the actual image tokens during processing.
            else:
                raise ValueError(f"Unsupported item type in interleaved sequence: {type(item)}")
        
        outputs = self.hf_processor(
            images=[images], # images is a batched to size of one.
            text=" ".join(texts),
            return_tensors="pt",
            images_kwargs={"do_rescale": False},
        )
        # Get tokens and images from processor output
        # Squeeze the batch dimension as we process one sample at a time
        tokens = outputs["input_ids"].squeeze(0)
        images = outputs.get("pixel_values")  # Use .get() for optional images

        # Convert images to bfloat16 and stack, or create an empty tensor if no images
        if images is not None and images.numel() > 0:
            # Ensure images tensor is on the same device as tokens/labels if needed
            images = images.to(device=tokens.device, dtype=torch.bfloat16)
            processed_images = images  # Already stacked by HF processor if multiple images/frames
        else:
            # Create an empty tensor with appropriate dimensions and dtype if no images
            processed_images = None
        return tokens, processed_images

    def compute_labels_interleaved(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute labels for an interleaved sample, ignoring image token IDs.

        This method generates a label tensor where the tokens corresponding to images are marked
        with the `ignore_place_holder` ID, and other tokens retain their original IDs.

        Parameters:
        tokens (torch.Tensor): A tensor containing the tokenized sequence.

        Returns:
        torch.Tensor: A tensor containing the labels for the tokenized sequence.
        """
        labels = tokens.clone()
        labels[labels == self.config.image_token_id] = self.config.ignore_place_holder
        labels = labels[1:].contiguous()
        return labels

    def pad_tokens_and_labels(self, tokens: torch.Tensor, labels: torch.Tensor, seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad tokens and labels to a be a multiple of config.pad_to_multiple_of

        Parameters:
        tokens (torch.Tensor): A tensor containing the tokenized sequence.
        labels (torch.Tensor): A tensor containing the labels for the tokenized sequence.
        seqlen (int): Original sequence length before padding

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tokens and labels tensor padded to a multiple of config.pad_to_multiple_of
        """
        seqlen_padded = (
                (seqlen + self.config.pad_to_multiple_of - 1)
                // self.config.pad_to_multiple_of
                * self.config.pad_to_multiple_of
            )
        pad_len = seqlen_padded - seqlen

        if pad_len > 0:
            tokens = F.pad(tokens, (0, pad_len), 'constant', 0)
            labels = F.pad(labels, (0, pad_len), 'constant', self.config.ignore_place_holder)
        return tokens, labels

    def compute_loss_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss mask based on which label values are negative.

        Parameters:
        labels (torch.Tensor): A tensor containing the labels for the tokenized sequence.

        Returns
        torch.Tensor: The computed loss mask.
        """

        loss_mask = torch.ones_like(labels, dtype=torch.float)
        loss_mask[labels < 0] = 0.0
        return loss_mask

    def encode_interleaved_sample(self, input_sample: InterleavedSample) -> DataSample:
        """
        Encode an interleaved sample.

        This method tokenizes the input sequence, computes labels and a loss mask, and processes
        the images. The encoded sample is then stored in the output_sample object.

        Parameters:
        input_sample (InterleavedSample): The interleaved sample to be encoded.

        Returns:
        DataSample: Encoded sample with processed image, tokens, labels and loss mask
        """
        logging.info(f"The config is: {self.config}")
        logging.info(f"input_sample={input_sample}")
        tokens, processed_images = self.tokenize_interleaved_sample(input_sample)

        logging.info(f"decode encoded tokens: {self.tokenizer.tokenizer.decode(tokens)}")

        # --- Label Generation ---
        labels = self.compute_labels_interleaved(tokens)

        logging.info(f"encoded:===== input_sample={input_sample}, tokens={tokens}, labels={labels}")
        # Prepare final tensors
        tokens = tokens[:-1]
        seqlen = len(tokens)  # Original sequence length before padding
        position_ids = torch.arange(seqlen, dtype=torch.int64)

        logging.debug(f"data encoder: position_ids = {position_ids}")

        # Pad tokens and labels to a multiple of `pad_to_multiple_of` if specified
        if self.config.pad_to_multiple_of:
            tokens, labels = self.pad_tokens_and_labels(tokens, labels, seqlen)

        # Compute loss mask
        loss_mask = self.compute_loss_mask(labels)

        logging.debug(f"There are {(labels > 0).sum()} valid labels.")

        sample = Gemma3DataSample(
            __key__=input_sample.__key__,
            __restore_key__=input_sample.__restore_key__,
            __subflavor__=input_sample.__subflavor__,
            __subflavors__=input_sample.__subflavors__,
            pixel_values=processed_images,
            input_ids=tokens,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
        )
        logging.debug(f"Gemma3 task encoder: sample: {sample.position_ids}")

        return sample
