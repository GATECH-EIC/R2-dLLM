# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
import copy
import datetime
import json
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def _write_jsonl_zst(path, rows):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required to write decode events as .jsonl.zst"
        ) from exc
    cctx = zstd.ZstdCompressor(level=3)
    with open(path, "wb") as raw:
        with cctx.stream_writer(raw) as writer:
            for row in rows:
                line = json.dumps(row, separators=(",", ":")).encode("utf-8")
                writer.write(line + b"\n")

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    factor=None,
    confidence_cluster_size=None,
    spatial_threshold=None,
    confidence_cluster_unmasked=None,
    token_cluster=None,
    temporal_steps=None,
    temporal_threshold=None,
    temporal_eval="ave",
    record_decode=False,
    decode_events_dir="decode_events_window",
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        confidence_cluster_size: Optional positive length for the sliding window confidence
            sweep that runs after the regular block decode completes. Disabled when not
            provided or when the value is non-positive.
        spatial_threshold: Average confidence threshold applied to the sliding window sweep.
            The sweep only runs when both this threshold and ``confidence_cluster_size``
            are set.
        confidence_cluster_unmasked: Maximum number of non-mask tokens permitted inside a
            window for it to be considered. Windows exceeding this count are skipped.
        token_cluster: Optional mode (``"confidence"``, ``"mid"``, or ``"random"``) that
            triggers an additional window scan after each decode step. When enabled, any
            window whose masked positions unanimously share the same top-1 token and
            contains a masked position whose top-1 confidence exceeds
            ``spatial_threshold`` will immediately decode one token in that window.
        temporal_steps: When positive (and together with ``temporal_threshold``),
            automatically force any still-masked position to the token that has held the
            top-1 prediction for this many consecutive decoding attempts.
        temporal_threshold: Average probability threshold that the consecutive streak must
            exceed before the position is auto-confirmed.
        temporal_eval: Strategy for aggregating confidences during auto-confirm.
            ``"ave"`` (default) requires the average confidence over the streak to exceed
            the threshold, ``"last"`` only checks the final step confidence, and ``"max"``
            uses the maximum confidence observed within the streak.
        record_decode: When True, dumps per-step decode artifacts to jsonl.zst files.
        decode_events_dir: Directory to write decode history files.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    token_confidence = torch.full(x.shape, float('-inf'), dtype=torch.float32, device=x.device)
    seq_len = x.shape[1]

    record_decode = bool(record_decode)
    if record_decode:
        all_xo_p = []
        all_transfer_index = []
        all_decoded_ids = []
        all_block_logits = []
        all_block_positions = []
        all_top_logit_ids = []
    else:
        all_xo_p = None
        all_transfer_index = None
        all_decoded_ids = None
        all_block_logits = None
        all_block_positions = None
        all_top_logit_ids = None

    window_size = 0
    if confidence_cluster_size is not None:
        try:
            window_size = int(confidence_cluster_size)
        except (TypeError, ValueError):
            window_size = 0
    if window_size <= 0:
        window_size = 0

    window_threshold = None
    if spatial_threshold is not None:
        try:
            window_threshold = float(spatial_threshold)
        except (TypeError, ValueError):
            window_threshold = None

    window_allow_unmasked_max = None
    if confidence_cluster_unmasked is not None:
        try:
            window_allow_unmasked_max = int(confidence_cluster_unmasked)
        except (TypeError, ValueError):
            window_allow_unmasked_max = None
        else:
            if window_allow_unmasked_max < 0:
                window_allow_unmasked_max = 0

    window_decode_enabled = window_size > 0 and window_threshold is not None

    temporal_consistency_mode = None
    if token_cluster is not None:
        temporal_consistency_mode = str(token_cluster).strip().lower()
        if not temporal_consistency_mode or temporal_consistency_mode == "none":
            temporal_consistency_mode = None
        elif temporal_consistency_mode not in {"confidence", "mid", "random"}:
            temporal_consistency_mode = None
    temporal_consistency_enabled = (
        temporal_consistency_mode is not None
        and window_size > 0
        and window_threshold is not None
    )
    temporal_consistency_dilate = 0

    valid_conf_eval_modes = {"ave", "last", "max"}
    if temporal_eval is None:
        auto_confirm_conf_eval_mode = "ave"
    else:
        auto_confirm_conf_eval_mode = str(temporal_eval).strip().lower()
        if auto_confirm_conf_eval_mode not in valid_conf_eval_modes:
            auto_confirm_conf_eval_mode = "ave"

    auto_confirm_steps = int(temporal_steps) if temporal_steps else 0
    if temporal_threshold is not None:
        try:
            auto_confirm_conf_threshold = float(temporal_threshold)
        except (TypeError, ValueError):
            auto_confirm_conf_threshold = None
    else:
        auto_confirm_conf_threshold = None

    auto_confirm_enabled = auto_confirm_steps > 0 and auto_confirm_conf_threshold is not None
    if auto_confirm_enabled:
        auto_confirm_counts = torch.zeros_like(x, dtype=torch.int32)
        auto_confirm_conf_sums = torch.zeros_like(x, dtype=torch.float32)
        auto_confirm_last_ids = torch.full_like(x, fill_value=-1, dtype=torch.long)
        auto_confirm_conf_max = (
            torch.zeros_like(x, dtype=torch.float32)
            if auto_confirm_conf_eval_mode == "max"
            else None
        )
    else:
        auto_confirm_counts = None
        auto_confirm_conf_sums = None
        auto_confirm_last_ids = None
        auto_confirm_conf_max = None

    def maybe_auto_confirm(logits, abs_start, abs_end, candidate_mask, active_mask):
        if not auto_confirm_enabled:
            return
        if logits is None or candidate_mask is None:
            return
        if abs_end <= abs_start:
            return
        slice_len = abs_end - abs_start
        if candidate_mask.shape[-1] != slice_len:
            raise ValueError(
                f"Auto confirm mask length mismatch ({candidate_mask.shape[-1]} vs {slice_len})."
            )
        candidate_mask = candidate_mask.to(device=x.device, dtype=torch.bool)
        if active_mask is None:
            active_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=x.device, dtype=torch.bool)
        active_mask = active_mask & candidate_mask
        counts_slice = auto_confirm_counts[:, abs_start:abs_end]
        sums_slice = auto_confirm_conf_sums[:, abs_start:abs_end]
        tracked_ids_slice = auto_confirm_last_ids[:, abs_start:abs_end]
        max_slice = (
            auto_confirm_conf_max[:, abs_start:abs_end]
            if auto_confirm_conf_eval_mode == "max"
            else None
        )
        if not candidate_mask.any():
            counts_slice.zero_()
            sums_slice.zero_()
            tracked_ids_slice.fill_(-1)
            if max_slice is not None:
                max_slice.zero_()
            return
        inactive_mask = candidate_mask & ~active_mask
        if inactive_mask.any():
            counts_slice[inactive_mask] = 0
            sums_slice[inactive_mask] = 0.0
            tracked_ids_slice[inactive_mask] = -1
            if max_slice is not None:
                max_slice[inactive_mask] = 0.0
        if not active_mask.any():
            return
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top_conf, top_ids = torch.max(probs, dim=-1)
        same_token_mask = active_mask & (tracked_ids_slice == top_ids)
        new_token_mask = active_mask & ~same_token_mask
        if same_token_mask.any():
            counts_slice[same_token_mask] = counts_slice[same_token_mask] + 1
            sums_slice[same_token_mask] = (
                sums_slice[same_token_mask] + top_conf[same_token_mask]
            )
            if max_slice is not None:
                max_slice[same_token_mask] = torch.maximum(
                    max_slice[same_token_mask],
                    top_conf[same_token_mask],
                )
        if new_token_mask.any():
            counts_slice[new_token_mask] = 1
            sums_slice[new_token_mask] = top_conf[new_token_mask]
            tracked_ids_slice[new_token_mask] = top_ids[new_token_mask]
            if max_slice is not None:
                max_slice[new_token_mask] = top_conf[new_token_mask]
        auto_confirm_counts[:, abs_start:abs_end] = counts_slice
        auto_confirm_conf_sums[:, abs_start:abs_end] = sums_slice
        auto_confirm_last_ids[:, abs_start:abs_end] = tracked_ids_slice
        if max_slice is not None:
            auto_confirm_conf_max[:, abs_start:abs_end] = max_slice
        confirm_mask = active_mask & (counts_slice >= auto_confirm_steps)
        if not confirm_mask.any():
            return
        if auto_confirm_conf_eval_mode == "ave":
            conf_map = sums_slice / counts_slice.clamp(min=1)
        elif auto_confirm_conf_eval_mode == "last":
            conf_map = top_conf
        else:
            conf_map = max_slice
        eval_values = conf_map[confirm_mask]
        eligible = eval_values >= auto_confirm_conf_threshold
        if not eligible.any():
            return
        final_mask = torch.zeros_like(confirm_mask)
        final_mask[confirm_mask] = eligible
        if not final_mask.any():
            return
        block_view = x[:, abs_start:abs_end]
        token_conf_view = token_confidence[:, abs_start:abs_end]
        confirmed_ids = tracked_ids_slice[final_mask]
        block_view[final_mask] = confirmed_ids
        token_conf_view[final_mask] = conf_map[final_mask].to(token_conf_view.dtype)
        counts_slice[final_mask] = 0
        sums_slice[final_mask] = 0.0
        tracked_ids_slice[final_mask] = -1
        if max_slice is not None:
            max_slice[final_mask] = 0.0
        auto_confirm_counts[:, abs_start:abs_end] = counts_slice
        auto_confirm_conf_sums[:, abs_start:abs_end] = sums_slice
        auto_confirm_last_ids[:, abs_start:abs_end] = tracked_ids_slice
        if max_slice is not None:
            auto_confirm_conf_max[:, abs_start:abs_end] = max_slice

    def _run_window_sweep(block_predictions, block_confidence, block_start, block_end):
        if (
            not window_decode_enabled
            or block_predictions is None
            or block_confidence is None
            or window_size <= 0
            or block_end <= block_start
            or window_threshold is None
        ):
            return False
        block_len = block_end - block_start
        if block_len < window_size:
            return False
        if block_predictions.shape[-1] != block_len:
            block_predictions = block_predictions[:, block_start:block_end]
        if block_confidence.shape[-1] != block_len:
            block_confidence = block_confidence[:, block_start:block_end]
        block_view = x[:, block_start:block_end]
        decode_mask = torch.zeros_like(block_view, dtype=torch.bool)
        max_start = block_len - window_size + 1
        for batch_idx in range(block_view.shape[0]):
            tokens_row = block_view[batch_idx]
            conf_row = block_confidence[batch_idx]
            if conf_row.shape[-1] != block_len:
                continue
            for start_idx in range(max_start):
                end_idx = start_idx + window_size
                window_tokens = tokens_row[start_idx:end_idx]
                if window_allow_unmasked_max is not None:
                    unmasked_count = (window_tokens != mask_id).sum().item()
                    if unmasked_count > window_allow_unmasked_max:
                        continue
                mask_positions = window_tokens == mask_id
                if not mask_positions.any():
                    continue
                window_conf = conf_row[start_idx:end_idx]
                masked_conf = window_conf[mask_positions]
                if masked_conf.numel() == 0:
                    continue
                if not torch.isfinite(masked_conf).all():
                    continue
                avg_conf = masked_conf.mean()
                if float(avg_conf.item()) >= window_threshold:
                    decode_slice = decode_mask[batch_idx, start_idx:end_idx]
                    decode_slice[mask_positions] = True
        if not decode_mask.any():
            return False
        block_view[decode_mask] = block_predictions[decode_mask]
        block_token_conf = token_confidence[:, block_start:block_end]
        conf_updates = block_confidence[decode_mask].to(block_token_conf.dtype)
        block_token_conf[decode_mask] = conf_updates
        return True

    def _apply_temporal_consistency(block_logits, block_start, block_end):
        if (
            not temporal_consistency_enabled
            or block_logits is None
            or block_end <= block_start
        ):
            return False
        block_len = block_end - block_start
        required_matches = 2
        if block_len < required_matches:
            return False
        logits_view = block_logits
        if logits_view.shape[1] != block_len:
            if logits_view.shape[1] >= block_len:
                logits_view = logits_view[:, :block_len, :]
            else:
                return False
        probs = F.softmax(logits_view.to(torch.float32), dim=-1)
        top_conf_vals, top_ids = torch.max(probs, dim=-1)
        block_view = x[:, block_start:block_end]
        block_conf_view = token_confidence[:, block_start:block_end]
        if temporal_consistency_dilate <= 0:
            decoded = False
            for batch_idx in range(block_view.shape[0]):
                tokens_row = block_view[batch_idx]
                conf_row = top_conf_vals[batch_idx]
                ids_row = top_ids[batch_idx]
                pos = 0
                while pos < block_len:
                    if int(tokens_row[pos].item()) != mask_id:
                        pos += 1
                        continue
                    streak_token_val = int(ids_row[pos].item())
                    streak_start = pos
                    pos += 1
                    while (
                        pos < block_len
                        and int(tokens_row[pos].item()) == mask_id
                        and int(ids_row[pos].item()) == streak_token_val
                    ):
                        pos += 1
                    streak_end = pos
                    streak_len = streak_end - streak_start
                    if streak_len < required_matches:
                        continue
                    conf_slice = conf_row[streak_start:streak_end]
                    if conf_slice.numel() == 0:
                        continue
                    max_conf_val, max_conf_idx = torch.max(conf_slice, dim=0)
                    if float(max_conf_val.item()) <= window_threshold:
                        continue
                    if temporal_consistency_mode == "confidence":
                        rel_choice = int(max_conf_idx.item())
                        conf_to_write = max_conf_val
                    elif temporal_consistency_mode == "random":
                        rand_idx = torch.randint(
                            0, streak_len, (1,), device=conf_slice.device
                        ).item()
                        rel_choice = int(rand_idx)
                        conf_to_write = conf_slice[rand_idx]
                    else:
                        rel_choice = streak_len // 2
                        conf_to_write = conf_slice[rel_choice]
                    block_rel_pos = streak_start + rel_choice
                    if block_rel_pos < 0 or block_rel_pos >= block_len:
                        continue
                    if int(tokens_row[block_rel_pos].item()) != mask_id:
                        continue
                    tokens_row[block_rel_pos] = streak_token_val
                    block_conf_view[batch_idx, block_rel_pos] = conf_to_write.to(
                        block_conf_view.dtype
                    )
                    decoded = True
            return decoded

        dilated_span = required_matches + temporal_consistency_dilate * max(
            required_matches - 1, 0
        )
        if dilated_span <= 0:
            dilated_span = required_matches
        decoded = False
        for batch_idx in range(block_view.shape[0]):
            tokens_row = block_view[batch_idx]
            conf_row = top_conf_vals[batch_idx]
            ids_row = top_ids[batch_idx]
            start_pos = 0
            while start_pos < block_len:
                if int(tokens_row[start_pos].item()) != mask_id:
                    start_pos += 1
                    continue
                window_end = min(block_len, start_pos + dilated_span)
                mask_slice = tokens_row[start_pos:window_end] == mask_id
                if int(mask_slice.sum().item()) < required_matches:
                    start_pos += 1
                    continue
                target_id = int(ids_row[start_pos].item())
                ids_slice = ids_row[start_pos:window_end]
                match_mask = mask_slice & (ids_slice == target_id)
                if int(match_mask.sum().item()) < required_matches:
                    start_pos += 1
                    continue
                conf_slice = conf_row[start_pos:window_end]
                match_positions = torch.nonzero(match_mask, as_tuple=False).squeeze(1)
                match_conf = conf_slice[match_mask]
                max_conf_val, max_conf_idx = torch.max(match_conf, dim=0)
                if float(max_conf_val.item()) <= window_threshold:
                    start_pos += 1
                    continue
                if temporal_consistency_mode == "confidence":
                    chosen_idx = int(max_conf_idx.item())
                    conf_to_write = max_conf_val
                elif temporal_consistency_mode == "random":
                    chosen_idx = torch.randint(
                        0, match_positions.shape[0], (1,), device=match_conf.device
                    ).item()
                    conf_to_write = match_conf[chosen_idx]
                else:
                    chosen_idx = match_positions.shape[0] // 2
                    conf_to_write = match_conf[chosen_idx]
                block_rel_pos = start_pos + int(match_positions[chosen_idx].item())
                if block_rel_pos < 0 or block_rel_pos >= block_len:
                    start_pos += 1
                    continue
                if int(tokens_row[block_rel_pos].item()) != mask_id:
                    start_pos += 1
                    continue
                tokens_row[block_rel_pos] = target_id
                block_conf_view[batch_idx, block_rel_pos] = conf_to_write.to(
                    block_conf_view.dtype
                )
                decoded = True
                start_pos += 1
        return decoded

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        block_candidate_mask = (
            mask_index[:, current_block_start:current_block_end].clone()
            if auto_confirm_enabled
            else None
        )
        if factor is None:
            x0, transfer_index, confidence = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
                return_confidence=True,
            )
        else:
            x0, transfer_index, confidence = get_transfer_index_dynamic(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                None,
                factor,
                return_confidence=True,
            )
        x[transfer_index] = x0[transfer_index]
        conf_view = confidence.to(token_confidence.dtype)
        token_confidence[transfer_index] = conf_view[transfer_index]
        _run_window_sweep(
            x0,
            confidence,
            current_block_start,
            current_block_end,
        )
        block_logits_view = output.logits[:, current_block_start:current_block_end, :]
        _apply_temporal_consistency(
            block_logits_view,
            current_block_start,
            current_block_end,
        )
        if auto_confirm_enabled:
            block_view = x[:, current_block_start:current_block_end]
            remaining_mask = block_candidate_mask & (block_view == mask_id)
            maybe_auto_confirm(
                block_logits_view,
                current_block_start,
                current_block_end,
                block_candidate_mask,
                remaining_mask,
            )
        if record_decode:
            p = F.softmax(output.logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1)
            all_xo_p.append(x0_p.detach().to(torch.float32).cpu().tolist())
            all_transfer_index.append(transfer_index.detach().cpu().tolist())
            all_decoded_ids.append(x.detach().cpu().tolist())
            block_start = prompt.shape[1] + num_block * block_length
            block_end = block_start + block_length
            all_block_logits.append(
                output.logits[:, block_start:block_end, :].detach().to(torch.float32).cpu().numpy()
            )
            all_block_positions.append({"block_start": int(block_start), "block_end": int(block_end)})
            top_ids = torch.argmax(output.logits, dim=-1)
            all_top_logit_ids.append(top_ids.detach().cpu().tolist())
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            block_candidate_mask = (
                mask_index.clone() if auto_confirm_enabled else None
            )
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index, confidence = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                    return_confidence=True,
                )
            else:
                x0, transfer_index, confidence = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    None,
                    factor,
                    return_confidence=True,
                )
            block_view = x[:, current_block_start:current_block_end]
            block_view[transfer_index] = x0[transfer_index]
            block_conf_view = token_confidence[:, current_block_start:current_block_end]
            block_conf_values = confidence.to(block_conf_view.dtype)
            block_conf_view[transfer_index] = block_conf_values[transfer_index]
            _run_window_sweep(
                x0,
                confidence,
                current_block_start,
                current_block_end,
            )
            _apply_temporal_consistency(
                logits,
                current_block_start,
                current_block_end,
            )
            if auto_confirm_enabled:
                remaining_mask = block_candidate_mask & (block_view == mask_id)
                maybe_auto_confirm(
                    logits,
                    current_block_start,
                    current_block_end,
                    block_candidate_mask,
                    remaining_mask,
                )
            if record_decode:
                top1_conf_full = torch.zeros(x.size(0), seq_len, device=x.device, dtype=torch.float32)

                p = F.softmax(logits.to(torch.float32), dim=-1)
                gather_idx = torch.unsqueeze(x[:, current_block_start:current_block_end], -1)
                x0_p_block = torch.squeeze(torch.gather(p, dim=-1, index=gather_idx), -1)

                top1_conf_full[:, current_block_start:current_block_end] = x0_p_block.detach().to(torch.float32)

                all_xo_p.append(top1_conf_full.detach().cpu().tolist())

                full_transfer = torch.zeros_like(x, dtype=torch.bool)
                full_transfer[:, current_block_start:current_block_end] = transfer_index
                all_transfer_index.append(full_transfer.cpu().tolist())
                all_decoded_ids.append(x.detach().cpu().tolist())
                all_block_logits.append(logits.detach().to(torch.float32).cpu().numpy())
                all_block_positions.append({"block_start": int(current_block_start), "block_end": int(current_block_end)})
                block_top_ids = torch.argmax(logits, dim=-1).detach()
                full_top_ids = copy.deepcopy(x)
                full_top_ids[:, current_block_start:current_block_end] = block_top_ids
                all_top_logit_ids.append(full_top_ids.cpu().tolist())
            i += 1

    if record_decode and all_xo_p:
        os.makedirs(decode_events_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        xo_p_filename = f"xo_p_steps_{timestamp}.jsonl.zst"
        transfer_index_filename = f"transfer_index_steps_{timestamp}.jsonl.zst"
        decoded_ids_filename = f"decoded_ids_steps_{timestamp}.jsonl.zst"
        top_ids_filename = f"top_ids_steps_{timestamp}.jsonl.zst"
        block_logits_filename = f"block_logits_steps_{timestamp}.npy"
        block_positions_filename = f"block_positions_steps_{timestamp}.jsonl.zst"
        xo_p_filepath = os.path.join(decode_events_dir, xo_p_filename)
        transfer_index_filepath = os.path.join(decode_events_dir, transfer_index_filename)
        decoded_ids_filepath = os.path.join(decode_events_dir, decoded_ids_filename)
        top_ids_filepath = os.path.join(decode_events_dir, top_ids_filename)
        block_logits_filepath = os.path.join(decode_events_dir, block_logits_filename)
        block_positions_filepath = os.path.join(decode_events_dir, block_positions_filename)
        _write_jsonl_zst(xo_p_filepath, all_xo_p)
        _write_jsonl_zst(transfer_index_filepath, all_transfer_index)
        _write_jsonl_zst(decoded_ids_filepath, all_decoded_ids)
        _write_jsonl_zst(top_ids_filepath, all_top_logit_ids)
        # if all_block_logits:
        #     block_logits_array = np.stack(all_block_logits, axis=0)
        #     np.save(block_logits_filepath, block_logits_array)
        #     with open(block_positions_filepath, "w") as f:
        #         json.dump(all_block_positions, f)

    return x, nfe


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
    return_confidence: bool = False,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    if return_confidence:
        return x0, transfer_index, confidence
    return x0, transfer_index

def get_transfer_index_dynamic(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
    return_confidence: bool = False,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    if return_confidence:
        return x0, transfer_index, confidence
    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
