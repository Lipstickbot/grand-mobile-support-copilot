"""Run and verify a pinned Soup local streaming experiment, with offline inference.

Run preflight first, then train, then reload. No cloud or publishing operations.
Detailed evidence and derived data stay under .private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.private'
SETUP = PRIVATE / 'soup-setup'
DATA = PRIVATE / 'soup-data'
RUN = PRIVATE / 'soup-runs/qwen3-8b-smoke'
CONFIG = ROOT / 'experiments/soup/smoke.yaml'


def environment():
    os.chdir(ROOT)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    for directory in ('hf-cache', 'torch-cache', 'soup-layer-cache', 'soup-setup'):
        (PRIVATE / directory).mkdir(exist_ok=True)
    os.environ.update({
        'HF_HOME': str(PRIVATE / 'hf-cache'),
        'HF_DATASETS_CACHE': str(PRIVATE / 'hf-cache/datasets'),
        'TORCH_HOME': str(PRIVATE / 'torch-cache'),
        'SOUP_LAYER_STREAM_CACHE_DIR': str(PRIVATE / 'soup-layer-cache'),
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'DO_NOT_TRACK': '1', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    })


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_identity():
    return {'train_sha256': file_sha(DATA / 'train64.jsonl'),
            'heldout_sha256': file_sha(DATA / 'heldout.jsonl'),
            'config_sha256': file_sha(CONFIG),
            'base_revision': json.loads((SETUP / 'qwen3-8b-metadata.json').read_text())['sha']}


def require_preflight_identity():
    prior = json.loads((SETUP / 'preflight.json').read_text(encoding='utf-8'))
    if prior.get('status') != 'passed':
        raise RuntimeError('A successful current preflight is required')
    for key, value in input_identity().items():
        if prior.get(key) != value:
            raise RuntimeError(f'{key} changed since successful preflight')
    for package, version in prior['versions'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'{package} changed since successful preflight')
    return prior


def make_subset():
    rows = read_rows(DATA / 'train.jsonl')
    quotas = {'insult': 44, 'cheats': 16, 'account': 2, 'payment': 2}
    selected, counts = [], Counter()
    for row in rows:
        topic = json.loads(row['messages'][-1]['content'])['topic']
        if counts[topic] < quotas[topic]:
            selected.append(row)
            counts[topic] += 1
    if len(selected) != 64 or dict(counts) != quotas:
        raise RuntimeError(f'Cannot form the fixed 64-example smoke subset: {counts}')
    path = DATA / 'train64.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    write_json(DATA / 'train64-manifest.json', {
        'source_sha256': file_sha(DATA / 'train.jsonl'), 'subset_sha256': file_sha(path),
        'selection': 'first examples per fixed topic quota; all from approved train split',
        'counts': dict(counts), 'rows': len(selected), 'gold': False,
        'scope': 'RU-only template/weak-target training mechanics smoke test'})
    return selected


def load_components():
    import torch
    from soup_cli.config.loader import load_config
    from transformers import AutoTokenizer
    cfg = load_config(CONFIG, raise_unreadable=True)
    if importlib.metadata.version('soup-cli') != '0.77.0':
        raise RuntimeError('This experiment was reviewed for Soup 0.77.0 only')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, local_files_only=True, trust_remote_code=False)
    return cfg, tokenizer, torch


def format_and_check(rows, tokenizer, cfg):
    from soup_cli.data.sft_format import build_format_row
    full_cfg = cfg.data.model_copy(update={'max_length': 32768})
    full_formatter = build_format_row(tokenizer, full_cfg, training_cfg=cfg.training)
    formatter = build_format_row(tokenizer, cfg.data, training_cfg=cfg.training)
    stats = []
    for index, row in enumerate(rows):
        full = full_formatter(row)
        formatted = formatter(row)
        if len(full['input_ids']) > cfg.data.max_length:
            raise RuntimeError(f'Example {index} exceeds token budget; do not silently truncate')
        if full['input_ids'] != formatted['input_ids'] or full['labels'] != formatted['labels']:
            raise RuntimeError(f'Example {index} changed under token-budget formatting')
        prefix = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                              add_generation_prompt=False, return_dict=False)
        if hasattr(prefix, 'tolist'):
            prefix = prefix.tolist()
        if full['input_ids'][:len(prefix)] != prefix:
            raise RuntimeError(f'Example {index} has a non-prefix-stable chat template')
        if any(label != -100 for label in formatted['labels'][:len(prefix)]):
            raise RuntimeError(f'Example {index} supervises a system/user prefix token')
        supervised = [token for token in formatted['labels'] if token != -100]
        if not supervised or len(supervised) == len(formatted['labels']):
            raise RuntimeError(f'Missing assistant-only mask for example {index}')
        decoded = tokenizer.decode(supervised, skip_special_tokens=False)
        target = row['messages'][-1]['content']
        if target not in decoded:
            raise RuntimeError(f'Assistant target not fully supervised for example {index}')
        # Detect user/system leakage by matching their content token subsequences.
        for message in row['messages'][:-1]:
            if message['content'] in decoded:
                raise RuntimeError(f'Input content leaked into assistant loss for example {index}')
        stats.append({'index': index, 'tokens': len(formatted['input_ids']),
                      'supervised_tokens': len(supervised)})
    return stats


def preflight():
    rows = make_subset()
    cfg, tokenizer, torch = load_components()
    heldout = read_rows(DATA / 'heldout.jsonl')
    stats = format_and_check(rows, tokenizer, cfg)
    val_stats = format_and_check(heldout, tokenizer, cfg)
    versions = {name: importlib.metadata.version(name) for name in
                ('soup-cli', 'torch', 'transformers', 'peft', 'trl', 'bitsandbytes',
                 'datasets', 'accelerate', 'safetensors', 'huggingface-hub')}
    result = {'status': 'passed', 'versions': versions, 'cuda_available': torch.cuda.is_available(),
              'train_count': len(rows), 'heldout_count': len(heldout),
              'expected_optimizer_steps': 64,
              'masking': 'all system/user prefix tokens masked; full assistant target retained',
              'max_train_tokens': max(s['tokens'] for s in stats),
              'max_heldout_tokens': max(s['tokens'] for s in val_stats),
              **input_identity(), 'rows': stats}
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(0), bf16=torch.cuda.is_bf16_supported(),
                      cuda_free_total=list(torch.cuda.mem_get_info()))
    write_json(SETUP / 'preflight.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, ensure_ascii=False), flush=True)


def probe_rows():
    rows = read_rows(DATA / 'heldout.jsonl')
    selected = []
    for topic in ('insult', 'cheats', 'account', 'payment'):
        selected.append(next(row for row in rows if json.loads(row['messages'][-1]['content'])['topic'] == topic))
    return selected


def forward_probe(wrapper, torch, *, disable_adapter=False):
    import contextlib
    from soup_cli.data.sft_format import build_format_row
    formatter = build_format_row(wrapper.tokenizer, wrapper.config.data,
                                 training_cfg=wrapper.config.training)
    model = wrapper.model
    was_training = model.training
    model.eval()
    losses, logits = [], []
    try:
        with (model.disable_adapter() if disable_adapter else contextlib.nullcontext()):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for row in probe_rows():
                    formatted = formatter(row)
                    batch = {key: torch.tensor([formatted[key]], device='cuda')
                             for key in ('input_ids', 'attention_mask', 'labels')}
                    outputs = model(**batch, use_cache=False)
                    loss = float(outputs.loss.detach().float().cpu())
                    if not math.isfinite(loss):
                        raise RuntimeError('Non-finite heldout probe loss')
                    losses.append(loss)
                    # Soup's fallback mask may supervise the role header too. Locate
                    # actual JSON content explicitly, rather than probe a header token.
                    target = row['messages'][-1]['content']
                    target_ids = wrapper.tokenizer.encode(target, add_special_tokens=False)
                    ids = formatted['input_ids']
                    matches = [i for i in range(len(ids) - len(target_ids) + 1)
                               if ids[i:i + len(target_ids)] == target_ids]
                    if len(matches) != 1:
                        raise RuntimeError('Cannot uniquely locate assistant JSON tokens for probe')
                    first = matches[0]
                    if formatted['labels'][first] == -100:
                        raise RuntimeError('Actual assistant JSON is masked out')
                    logits.append(outputs.logits[0, max(0, first - 1)].detach().float().cpu().contiguous())
                    del outputs, batch
        stacked = torch.stack(logits)
        if not bool(torch.isfinite(stacked).all()):
            raise RuntimeError('Nonfinite probe logits')
        return {'mean_loss': sum(losses) / len(losses), 'losses': losses,
                'prediction_position': 'first token of actual assistant JSON'}, stacked
    finally:
        model.train(was_training)


def wrapper_setup():
    from soup_cli.trainer.sft import SFTTrainerWrapper
    cfg, tokenizer, torch = load_components()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing accidental CPU 8B training')
    manifest = ROOT / cfg.base / 'download-manifest.json'
    if not manifest.is_file():
        raise RuntimeError('Verified complete author-model download is required')
    manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
    if manifest_data['revision'] != input_identity()['base_revision']:
        raise RuntimeError('Downloaded base revision differs from preflight source')
    rows = read_rows(DATA / 'train64.jsonl')
    if len(rows) != 64:
        raise RuntimeError('Expected exactly 64 training rows')
    format_and_check(rows, tokenizer, cfg)
    wrapper = SFTTrainerWrapper(cfg, device='cuda', report_to='none', trust_remote_code=False)
    wrapper.setup({'train': rows})
    return wrapper, torch


def train():
    from safetensors.torch import save_file
    from transformers import TrainerCallback
    require_preflight_identity()
    if (RUN / 'adapter_model.safetensors').exists():
        raise RuntimeError('Existing adapter preserved; select a new output directory for another run')
    started = time.time()
    wrapper, torch = wrapper_setup()
    evidence = {'scope': '64-step RU weak/template smoke; not human-gold quality',
                **input_identity()}
    before, before_logits = forward_probe(wrapper, torch, disable_adapter=True)
    RUN.mkdir(parents=True, exist_ok=True)
    save_file({'logits': before_logits}, str(RUN / 'base-probe.safetensors'))
    evidence['before'] = before
    write_json(RUN / 'experiment.json', evidence)
    # Observe the actual tensor returned from each training step, before HF's
    # logging_nan_inf_filter can hide a bad loss behind a finite average.
    raw_steps = []
    original_training_step = wrapper.trainer.training_step
    wrapper.trainer.args.logging_nan_inf_filter = False

    def checked_training_step(*args, **kwargs):
        loss = original_training_step(*args, **kwargs)
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError('Nonfinite raw training-step loss')
        for name, parameter in wrapper.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f'Nonfinite training gradient: {name}')
        raw_steps.append(float(loss.detach().float().cpu()))
        return loss

    wrapper.trainer.training_step = checked_training_step

    class EvidenceCallback(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            after, after_logits = forward_probe(wrapper, torch)
            base_again, disabled_logits = forward_probe(wrapper, torch, disable_adapter=True)
            evidence.update(after=after, disabled_after=base_again,
                            logit_max_abs_delta=float((after_logits - disabled_logits).abs().max()),
                            base_repeat_max_abs_delta=float((before_logits - disabled_logits).abs().max()),
                            observed_steps=state.global_step,
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            if not torch.allclose(before_logits, disabled_logits, atol=0.02, rtol=0.005):
                raise RuntimeError('Adapter-disabled base predictions drifted unexpectedly')
            if evidence['logit_max_abs_delta'] <= max(1e-4, 3 * evidence['base_repeat_max_abs_delta']):
                raise RuntimeError('No adapter prediction effect above base-repeat noise')
            save_file({'logits': after_logits}, str(RUN / 'trained-probe.safetensors'))
            write_json(RUN / 'experiment.json', evidence)

    wrapper.trainer.add_callback(EvidenceCallback())
    result = wrapper.train()
    evidence.update(training_result=result, wall_seconds=round(time.time() - started, 3))
    if wrapper.trainer.state.global_step != 64:
        raise RuntimeError(f'Unexpected actual step count: {wrapper.trainer.state.global_step}')
    if len(raw_steps) != 64:
        raise RuntimeError(f'Expected 64 checked raw training steps, observed {len(raw_steps)}')
    evidence['raw_step_losses'] = raw_steps
    evidence['all_64_raw_losses_and_observed_gradients_finite'] = True
    for row in wrapper.trainer.state.log_history:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f'Non-finite training metric {key}')
    wrapper.trainer.save_state()
    evidence['training_status'] = 'complete'
    write_json(RUN / 'experiment.json', evidence)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


def reload():
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    require_preflight_identity()
    experiment = json.loads((RUN / 'experiment.json').read_text(encoding='utf-8'))
    if experiment.get('training_status') != 'complete' or experiment.get('observed_steps') != 64:
        raise RuntimeError('Only a successful complete 64-step run can pass reload verification')
    for key, value in input_identity().items():
        if experiment.get(key) != value:
            raise RuntimeError(f'{key} changed since training')
    wrapper, torch = wrapper_setup()
    try:
        saved = load_file(str(RUN / 'adapter_model.safetensors'), device='cpu')
        if not saved or any('.inner.' in name or '_orig_mod.' in name for name in saved):
            raise RuntimeError('Empty or nonportable adapter keys')
        if not all(bool(torch.isfinite(value).all()) for value in saved.values()):
            raise RuntimeError('Nonfinite adapter tensor')
        b_tensors = [value for name, value in saved.items() if 'lora_B' in name]
        if not b_tensors or not any(bool(torch.count_nonzero(value)) for value in b_tensors):
            raise RuntimeError('Adapter is unchanged from zero-B initialization')
        set_peft_model_state_dict(wrapper.model, saved, adapter_name='default')
        loaded = get_peft_model_state_dict(wrapper.model, adapter_name='default')
        if set(loaded) != set(saved):
            raise RuntimeError('Reloaded adapter key mismatch')
        for name, expected in saved.items():
            actual = loaded[name].detach().cpu().to(expected.dtype)
            if not torch.equal(actual, expected):
                raise RuntimeError(f'Reloaded tensor mismatch: {name}')
        after, logits = forward_probe(wrapper, torch)
        expected_logits = load_file(str(RUN / 'trained-probe.safetensors'))['logits']
        max_delta = float((logits - expected_logits).abs().max())
        result = {'adapter_tensors': len(saved), 'nonzero_lora_B_tensors':
                  sum(bool(torch.count_nonzero(value)) for value in b_tensors),
                  'all_saved_tensors_reloaded_exactly': True,
                  'reloaded_probe': after, 'reload_max_abs_logit_delta': max_delta,
                  'functional_reload_close': bool(torch.allclose(logits, expected_logits, atol=0.02, rtol=0.005)),
                  'generated_text_evaluation': 'NOT RUN'}
        write_json(RUN / 'reload-verification.json', result)
        if not result['functional_reload_close']:
            raise RuntimeError('Reloaded model predictions differ from saved trained reference')
        print(json.dumps(result), flush=True)
    finally:
        wrapper._close_stream_runtime()


def main():
    environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'train', 'reload'))
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as exc:
        if args.stage == 'preflight':
            write_json(SETUP / 'preflight.json', {'status': 'failed', 'message': str(exc)})
        write_json(SETUP / f'{args.stage}-error.json', {'stage': args.stage, 'type': type(exc).__name__,
                                                    'message': str(exc), 'status': 'failed'})
        raise


if __name__ == '__main__':
    main()
