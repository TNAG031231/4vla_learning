from __future__ import annotations

import torch
from torch import nn

from src.phase0.drivema_stage2_inference import _batch_inputs, _generated_ids
from src.phase0.drivema_stage2_protocol import parse_trajectory
from src.phase0.phase0_4b_protocol import SFTSample


def intervention_action_ids(processor: object, sample: SFTSample) -> list[int]:
    tokenizer = processor.tokenizer
    ids = tokenizer(sample.target.training_text(), add_special_tokens=False)['input_ids']
    return [*ids, tokenizer.eos_token_id]


@torch.no_grad()
def replay_turn2(model: nn.Module, processor: object, first_inputs: dict, saved: dict,
                 action_ids: list[int], max_new_tokens: int) -> dict:
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token_id
    action, trajectory = saved['assistant_spans']
    ids = saved['input_ids']
    prefix = first_inputs['input_ids'][0].tolist()
    if (action['turn'], trajectory['turn']) != (1, 2) or not (
        len(prefix) == action['start'] < action['end'] < trajectory['start'] < trajectory['end'] == len(ids)
    ):
        raise ValueError(f"invalid saved two-turn spans for {saved['sample_token']}")
    if ids[:action['start']] != prefix:
        raise ValueError(f"observation prefix differs for {saved['sample_token']}")
    for span in (action, trajectory):
        if ids[span['start']:span['end']] != span['token_ids'] or span['token_ids'][-1] != eos:
            raise ValueError(f"saved token span or EOS differs for {saved['sample_token']}")
    if not action_ids or action_ids[-1] != eos:
        raise ValueError('replacement action must include EOS')

    suffix = ids[action['end']:trajectory['start']]
    context = first_inputs['input_ids'].new_tensor([prefix + action_ids + suffix])
    inputs = _batch_inputs([{**first_inputs, 'input_ids': context}], tokenizer.pad_token_id)
    model.eval()
    sequence = model.generate(**inputs, do_sample=False, num_beams=1,
                              max_new_tokens=max_new_tokens, use_cache=True)
    generation_eos = model.generation_config.eos_token_id
    stops = set(generation_eos if isinstance(generation_eos, list) else [generation_eos])
    stops.add(eos)
    generated = _generated_ids(sequence[0], context.shape[1], stops)
    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    complete = bool(generated) and generated[-1] == eos
    points = parse_trajectory(text) if complete else None
    reason = ('unterminated_trajectory' if not complete else
              'invalid_trajectory_text' if points is None else None)
    return {'trajectory_ids': generated, 'trajectory_text': text, 'waypoints': points, 'invalid_reason': reason}
