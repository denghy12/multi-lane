"""Compare trained uniform CLS exchange with post-projection Q/K normalization."""
import argparse
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import run_r1_final_cls_exchange as base
from multi_lane.track_a.evaluation_scores import load_evaluation_scores
from multi_lane.track_a.r1_crossvit import atomic_save, rng_state, restore_rng
from multi_lane.track_a.r1_crossvit_experiment import read, write, model_from_config
from multi_lane.track_a.runner import git_metadata, set_seed, task_indices, compute_metrics, summarize_tasks

MODES = ('uniform', 'normalized')


def digest(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


class Readout(base.Exchange):
    def __init__(self, width, mode):
        super().__init__(width, 'attention')
        self.mode = mode

    def components(self, features, reliable):
        tokens = self.project(features) + self.view_embedding
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        valid[:, 2] = reliable
        tokens = tokens.masked_fill(~valid[..., None], 0)
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        allowed = ~torch.eye(3, dtype=torch.bool, device=tokens.device) & valid[:, None, :]
        if self.mode == 'normalized':
            query = F.layer_norm(query, (query.shape[-1],))
            key = F.layer_norm(key, (key.shape[-1],))
        scores = query @ key.transpose(-1, -2) / query.shape[-1] ** .5
        if self.mode == 'uniform':
            weights = allowed.to(value.dtype) / allowed.sum(-1, keepdim=True)
        else:
            weights = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
        return weights, value, scores

    def forward(self, features, reliable):
        weights, value, _ = self.components(features, reliable)
        delta = self.output(weights @ value)
        return delta * (base.CAP / delta.norm(dim=-1, keepdim=True).clamp_min(base.CAP))


def gradients(module):
    for parameter in module.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Non-finite interaction gradient')
    grad = module.qkv.weight.grad
    size = grad.shape[0] // 3
    return {'query': float(grad[:size].norm()), 'key': float(grad[size:2*size].norm()),
            'value': float(grad[2*size:].norm()), 'output': float(module.output.weight.grad.norm())}


@torch.no_grad()
def diagnostic(module, features, reliable):
    weights, _, scores = module.components(features, reliable)
    delta = module(features, reliable)
    result = {}
    for receiver, name in ((1, 'person'), (2, 'face')):
        senders = [i for i in range(3) if i != receiver]
        probabilities = weights[reliable, receiver][:, senders]
        gap = scores[reliable, receiver, senders[0]] - scores[reliable, receiver, senders[1]]
        active = reliable if receiver == 2 else torch.ones_like(reliable)
        norms = delta[active, receiver].norm(dim=-1)
        result[name] = dict(full_weight=float(probabilities[:, 0].mean()),
                            weight_std=float(probabilities[:, 0].std(unbiased=False)),
                            score_gap_rms=float(gap.square().mean().sqrt()),
                            normalized_entropy=float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean() / np.log(2)),
                            mean_delta_norm=float(norms.mean()),
                            cap_fraction=float((norms >= base.CAP - 1e-5).float().mean()))
    return result


def fixed_indices(batch):
    reliable = torch.where(batch.reliable)[0][:32]
    if len(reliable) < 2:
        raise ValueError('TRAIN probe needs at least two reliable Face samples')
    unreliable = torch.where(~batch.reliable)[0][:32]
    return torch.cat([reliable, unreliable])


def train_probe(initial, batch, heads):
    index = fixed_indices(batch)
    x = batch.features[index, :, 0].cuda()
    reliable = batch.reliable[index].cuda()
    logits, labels, weights = [item[index].cuda() for item in (batch.logits, batch.targets, batch.weights)]
    checks = {}
    for mode in MODES:
        module = Readout(x.shape[-1], mode).cuda()
        module.load_state_dict(initial)
        optimizer = torch.optim.Adam(module.parameters(), lr=.0004)
        steps = []
        if not torch.equal(base.fuse(logits, weights), base.fuse(logits + base.correction(module, x, reliable, heads), weights)):
            raise AssertionError('Zero initialization changed original logits')
        for step in range(3):
            prediction = base.fuse(logits + base.correction(module, x, reliable, heads), weights)
            loss = F.binary_cross_entropy_with_logits(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            grad = gradients(module)
            if step == 0 and any(grad[key] != 0 for key in ('query', 'key', 'value')):
                raise AssertionError('Unexpected gradient through zero output initialization')
            if step > 0 and (grad['value'] <= 0 or grad['output'] <= 0):
                raise AssertionError('Value/output path has no gradient')
            if mode == 'uniform' and (grad['query'] != 0 or grad['key'] != 0):
                raise AssertionError('Uniform reader updated Q/K')
            if mode == 'normalized' and step > 0 and min(grad['query'], grad['key']) <= 0:
                raise AssertionError('Normalized Q/K has no gradient')
            steps.append(dict(step=step, loss=float(loss.detach()), gradients=grad,
                              diagnostics=diagnostic(module, x, reliable)))
            optimizer.step()
        checks[mode] = steps
    return dict(sample_ids=[batch.ids[i] for i in index.tolist()], checks=checks)


def fit(mode, batch, stage, heads, initial, output, epochs, protocol):
    folder = output / mode / f'task{stage}'
    folder.mkdir(parents=True, exist_ok=True)
    module = Readout(batch.features.shape[-1], mode).cuda()
    module.load_state_dict(initial)
    optimizer = torch.optim.Adam(module.parameters(), lr=.0004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator().manual_seed(1901 + stage)
    history, first_epoch = [], 0
    completed = sorted(folder.glob('epoch*.pth'))
    if completed:
        saved = torch.load(completed[-1], map_location='cpu', weights_only=False)
        if saved['protocol'] != protocol:
            raise ValueError('Recovery protocol differs')
        module.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['shuffle_rng'])
        restore_rng(saved['rng'])
        history, first_epoch = saved['history'], saved['epoch']
    x, y, z, reliable, weights = [item.cuda() for item in
        (batch.features[:, :, 0], batch.targets, batch.logits, batch.reliable, batch.weights)]
    probe = fixed_indices(batch).cuda()
    for epoch in range(first_epoch, epochs):
        loss_sum, samples, grad_sum, updates = 0., 0, dict(query=0., key=0., value=0., output=0.), 0
        for index in torch.randperm(len(x), generator=generator).split(64):
            index = index.cuda()
            logits = base.fuse(z[index] + base.correction(module, x[index], reliable[index], heads), weights[index])
            loss = F.binary_cross_entropy_with_logits(logits, y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite TRAIN loss')
            optimizer.zero_grad()
            loss.backward()
            grad = gradients(module)
            for key, value in grad.items():
                grad_sum[key] += value
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            samples += len(index)
            updates += 1
        scheduler.step()
        if not all(torch.isfinite(value).all() for value in module.state_dict().values()):
            raise FloatingPointError('Non-finite trained parameter')
        if mode == 'uniform' and not torch.equal(module.qkv.weight[:128].detach().cpu(), initial['qkv.weight'][:128].cpu()):
            raise AssertionError('Unused uniform Q/K changed')
        row = dict(epoch=epoch+1, train_BCE=loss_sum/samples, updates=updates,
                   gradient_means={key: value/updates for key, value in grad_sum.items()},
                   diagnostics=diagnostic(module, x[probe], reliable[probe]))
        history.append(row)
        atomic_save(dict(protocol=protocol, epoch=epoch+1, model=module.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         shuffle_rng=generator.get_state(), rng=rng_state(),
                         initialization_sha256=digest(initial), history=history), folder/f'epoch{epoch+1:02d}.pth')
        write(folder/'history.json', history)
        print('FIT', stage, mode, epoch+1, row['train_BCE'], flush=True)
    module.requires_grad_(False).eval()
    return module


def cache_batch(root, stage, split, models, sources, smoke):
    path = root/'features'/split/f'task{stage}.pth'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    batch = base.extract(models, sources, stage, split, smoke)
    atomic_save(batch, path)
    return batch


def save_scores(path, payload):
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['sample_ids'] != payload['sample_ids'] or any(
                not torch.equal(saved[key], payload[key]) for key in ('logits', 'targets', 'reliable')):
            raise ValueError(f'Existing prediction differs: {path}')
        return
    atomic_save(payload, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    git = git_metadata(source)
    if git['dirty']:
        raise ValueError('Clean Git source required')
    paths = [args.base_root/'native'/view/'train' for view in base.VIEWS]
    base_summary = read(args.base_root/'completion.json')
    if base_summary['status'] != 'complete' or base_summary['smoke']:
        raise ValueError('A completed formal native training stage is required')
    if any(len(list((folder/'compact_checkpoints').glob('task*.pth'))) != 8 for folder in paths):
        raise ValueError('Eight native checkpoints per view are required')
    inputs = dict(clip_checkpoint=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'),
                  data_root=Path('../multi-lane-main/datasets/EMOTIC'),
                  face_manifest_root=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in paths for path in [folder/'config.json', *sorted((folder/'compact_checkpoints').glob('task*.pth'))]}
    protocol = dict(git=git, base_root=str(args.base_root), source_hashes=hashes,
                    smoke=args.smoke, seed=1, epochs=1 if args.smoke else 5, batch_size=64,
                    learning_rate=.0004, cap=base.CAP, modes=list(MODES),
                    train_total_parameters=78016, uniform_effective_parameters=69824,
                    qk_normalization='separate post-projection LayerNorm, no affine',
                    primary_fusion='fixed_R1_logits_then_sigmoid', inputs={k:str(v) for k,v in inputs.items()},
                    advance_rule='Val final and average exceed both trained uniform and native R1')
    protocol['face_manifest_hashes'] = {
        split: hashlib.sha256((inputs['face_manifest_root']/'manifests'/f'{split}.jsonl').read_bytes()).hexdigest()
        for split in ('train', 'val', 'test')
    }
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Existing output has a different protocol')
    write(protocol_path, protocol)
    configs = [read(folder/'config.json') for folder in paths]
    if any(c['seed'] != 1 or c['r1_kind'] != 'native' for c in configs):
        raise ValueError('Expected common seed1 native R1 stage')
    set_seed(1, True)
    models = [model_from_config(config, inputs['clip_checkpoint']).eval().requires_grad_(False) for config in configs]
    torch.set_num_threads(2)
    inputs = base.SimpleNamespace(**inputs)
    splits = ('train','val') if args.smoke else ('train','val','test')
    sources = {split: [base.dataset(c, split, inputs) for c in configs] for split in splits}
    modules, heads = {mode:[] for mode in MODES}, []
    rows = {split:{mode:[[],[],[]] for mode in ('native','patch_pool',*MODES,'normalized_uniform')}
            for split in splits[1:]}
    for stage in range(2 if args.smoke else 8):
        for model, folder in zip(models, paths):
            saved = torch.load(folder/'compact_checkpoints'/f'task{stage}.pth', map_location='cpu', weights_only=False)
            missing = model.load_state_dict(saved['model'], strict=False)
            if missing.unexpected_keys or any(not key.startswith('visual_encoder.') for key in missing.missing_keys):
                raise ValueError('Incomplete base state')
            model.restore_task(stage)
            model.requires_grad_(False)
        for old_task, old_heads in enumerate(heads):
            current = torch.stack([m.head.weight[list(task_indices(old_task))] for m in models])
            if not torch.equal(old_heads, current):
                raise ValueError('An old classifier block changed')
        heads.append(torch.stack([m.head.weight[list(task_indices(stage))] for m in models]))
        training = cache_batch(args.output, stage, 'train', models, sources['train'], args.smoke)
        torch.manual_seed(1001+stage)
        initial = {key:value.detach().cpu().clone() for key,value in Readout(512,'normalized').state_dict().items()}
        checks = train_probe(initial, training, heads[-1])
        checks['initialization_sha256'] = digest(initial)
        write(args.output/'preflight'/f'task{stage}.json', checks)
        print('PREFLIGHT_PASS', stage, checks['initialization_sha256'], flush=True)
        previous = {mode:[digest(m.state_dict()) for m in bank] for mode,bank in modules.items()}
        for mode in MODES:
            modules[mode].append(fit(mode, training, stage, heads[-1], initial, args.output,
                                     protocol['epochs'], protocol))
        if any(previous[mode] != [digest(m.state_dict()) for m in modules[mode][:-1]] for mode in MODES):
            raise AssertionError('An old interaction module changed')
        del training
        for split in splits[1:]:
            batch = cache_batch(args.output, stage, split, models, sources[split], args.smoke)
            score_dir = 'val/scores' if split=='val' else 'train/test_scores'
            def reference_logits(kind, view):
                dump = load_evaluation_scores(args.reference/kind/view/score_dir/f'task{stage}.npz',64)
                positions = {sample:i for i,sample in enumerate(dump.sample_ids)}
                order = [positions[sample] for sample in batch.ids]
                if not np.array_equal(dump.targets[order], batch.targets.numpy()):
                    raise ValueError('Reference labels differ')
                return torch.from_numpy(dump.logits[order])
            for index,view in enumerate(base.VIEWS):
                if not torch.equal(reference_logits('native',view),batch.logits[:,index]):
                    raise ValueError(f'Native replay failed: {split}, {view}, task{stage}')
            scores = {'native':base.fuse(batch.logits,batch.weights)}
            patch = batch.logits.clone();patch[:,2] = reference_logits('pool','face')
            scores['patch_pool'] = base.fuse(patch,batch.weights)
            scores.update({mode:base.score(batch,modules[mode],heads) for mode in MODES})
            uniform_override = copy.deepcopy(modules['normalized'])
            for module in uniform_override:
                module.mode = 'uniform'
            scores['normalized_uniform'] = base.score(batch, uniform_override, heads)
            for mode,logits in scores.items():
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Non-finite evaluation logits')
                for group,mask in enumerate((torch.ones_like(batch.reliable),batch.reliable,~batch.reliable)):
                    if mask.any():
                        rows[split][mode][group].append(compute_metrics(stage,logits[mask].sigmoid(),batch.targets[mask],.5))
                save_scores(args.output/'scores'/split/mode/f'task{stage}.pth',
                            dict(logits=logits,targets=batch.targets,sample_ids=batch.ids,
                                 reliable=batch.reliable))
                print('EVAL',stage,split,mode,rows[split][mode][0][-1].mAP,flush=True)
            del batch
        write(args.output/'progress.json',dict(stage=stage,status='running',updated_unix=time.time()))
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value for path,value in hashes.items()):
        raise AssertionError('Original R1 checkpoint changed')
    results = {split:{mode:{name:dict(summary=summarize_tasks(values),stages=[r.mAP for r in values],
                                     per_class_ap=[r.per_class_ap for r in values])
                           for name,values in zip(('all','reliable','unreliable'),groups) if values}
                     for mode,groups in arms.items()} for split,arms in rows.items()}
    val = results['val']
    advance = all(val['normalized']['all']['summary'][key] > val[reference]['all']['summary'][key]
                  for key in ('final_mAP', 'average_mAP') for reference in ('uniform', 'native'))
    write(args.output/'completion.json',dict(status='complete',smoke=args.smoke,
          protocol=protocol,base_replay_exact=True,source_hashes_unchanged=True,
          advance_receivers_on_validation=advance, results=results))


if __name__ == '__main__':
    main()
