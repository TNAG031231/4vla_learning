from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import random

from PIL import Image
import torch

from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.phase0_4b_protocol import SFTSample, inference_messages
from src.phase0.qwen3vl_smoke import resolve_image_path
from src.phase0.phase0_4c_full_train import write_predictions

CONDITIONS = ('gray', 'scene_shuffle', 'history_gray', 'current_gray')


def donor_mapping(samples: list[SFTSample], donors: list[SFTSample], seed: int) -> dict:
    if any(s.split != 'validation' for s in samples + donors):
        raise ValueError('visual ablation only permits validation donors and sources')
    rng = random.Random(seed)
    ordered = sorted(donors, key=lambda s: s.sample_token)
    mapping = {}
    for sample in samples:
        candidates = [s for s in ordered if s.scene_token != sample.scene_token]
        if not candidates:
            raise ValueError('scene shuffle requires a different validation scene')
        slots = [i for i, valid in enumerate(sample.observation.history_valid_mask) if valid]
        replacements = []
        for image_index, slot in enumerate(slots):
            donor = rng.choice(candidates)
            donor_index = rng.randrange(len(donor.observation.image_paths))
            donor_slots = [i for i, valid in enumerate(donor.observation.history_valid_mask) if valid]
            replacements.append({
                'source_sample_token': sample.sample_token, 'source_scene_token': sample.scene_token,
                'source_image_index': image_index, 'source_history_slot': slot,
                'source_path': sample.observation.image_paths[image_index],
                'donor_sample_token': donor.sample_token, 'donor_scene_token': donor.scene_token,
                'donor_image_index': donor_index, 'donor_history_slot': donor_slots[donor_index],
                'donor_path': donor.observation.image_paths[donor_index], 'donor_split': donor.split,
            })
        mapping[sample.sample_token] = replacements
    return mapping


def intervene(images: list[Image.Image], condition: str, gray_rgb: tuple[int, int, int],
              replacements: list[dict], load_donor: Callable[[str], Image.Image]) -> tuple[list[Image.Image], list[int]]:
    if condition not in CONDITIONS:
        raise ValueError('unknown visual intervention')
    indices = (list(range(len(images) - 1)) if condition == 'history_gray' else
               [len(images) - 1] if condition == 'current_gray' else list(range(len(images))))
    output = list(images)
    for index in indices:
        size = images[index].size
        if condition == 'scene_shuffle':
            output[index] = load_donor(replacements[index]['donor_path']).resize(size, Image.Resampling.BICUBIC)
        else:
            output[index] = Image.new('RGB', size, gray_rgb)
    return output, indices


class VisualAblationRunner(Stage2Runner):
    def configure(self, condition: str, mapping: dict, gray_rgb: tuple[int, int, int],
                  clean: list[dict], evidence_path: Path) -> None:
        self.condition = condition
        self.mapping = mapping
        self.gray_rgb = gray_rgb
        self.clean = {row['sample_token']: row for row in clean}
        self.evidence_path = evidence_path
        self.evidence = {}

    def images(self, sample: SFTSample) -> list[Image.Image]:
        original = super().images(sample)
        changed, indices = intervene(
            original, self.condition, self.gray_rgb, self.mapping[sample.sample_token],
            lambda path: self.collator.image_loader(resolve_image_path(self.collator.dataset_root, path)),
        )
        encoded = [self.processor.apply_chat_template(
            inference_messages(sample.observation, images), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
        ) for images in (original, changed)]
        before, after = encoded
        same = {key: torch.equal(before[key], after[key])
                for key in ('input_ids', 'attention_mask', 'image_grid_thw')}
        same['pixel_values_shape'] = before['pixel_values'].shape == after['pixel_values'].shape
        clean = self.clean[sample.sample_token]
        prefix = clean['input_ids'][:clean['assistant_spans'][0]['start']]
        same['saved_clean_prefix'] = before['input_ids'][0].tolist() == prefix
        image_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        slots = [i for i, valid in enumerate(sample.observation.history_valid_mask) if valid]
        evidence = {
            'sample_token': sample.sample_token, 'condition': self.condition,
            'applicable': bool(indices), 'intervened_image_indices': indices,
            'intervened_history_slots': [slots[i] for i in indices],
            'history_valid_mask': list(sample.observation.history_valid_mask),
            'image_count': len(changed), 'image_sizes': [list(image.size) for image in changed],
            'structure_matches': same,
            'image_grid_thw': {name: value['image_grid_thw'].tolist()
                               for name, value in zip(('clean', 'perturbed'), encoded)},
            'image_token_positions': {name: value['input_ids'][0].eq(image_id).nonzero().flatten().tolist()
                                      for name, value in zip(('clean', 'perturbed'), encoded)},
        }
        self.evidence[sample.sample_token] = evidence
        # Persist the failed comparison too, before stopping inference.
        write_predictions(self.evidence_path, list(self.evidence.values()))
        if not all(same.values()):
            raise ValueError(f'visual input structure or saved clean prefix differs: {sample.sample_token}: {same}')
        return changed
