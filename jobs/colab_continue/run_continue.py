"""Resume the original signed trainer; operational monitoring lives outside it."""
from pathlib import Path
import gc
import json
import math
import os
import signal
import sys
import time
import traceback

os.environ['USE_TF'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drive_store import DriveStore
from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.data import ExperimentBudget, atomic_json, prepare_data
from recsys.two_tower_qwen.training import Trainer, BudgetExpired, make_model, select_device
from recsys.two_tower_qwen.model import _cached_pretrained
from recsys.two_tower_qwen.bundle import export_bundle
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd


def main():
    root = Path(os.environ.get('RECSYS_RUNTIME_ROOT', '/content/qwen_continue'))
    settings = json.loads((root / 'launch.json').read_text())
    code, work = root / 'payload', root / 'payload/work'
    artifacts = work / 'artifacts'
    reports = work / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    source = artifacts / 'runs' / settings['signature'][:20]
    original = torch.load(source / 'last.pt', map_location='cpu', weights_only=True)
    assert original['signature'] == settings['signature'] and original['step'] == settings['initial_step']
    baseline_seconds = original['elapsed_seconds']
    baseline_progress = original['epoch'] + original['cursor'] / settings['fit_queries']
    target = baseline_progress + 5
    store = DriveStore(root / 'drive_credentials.json', settings['folder_id'], settings['quota_project'])
    started = time.monotonic()
    available = settings['deadline_unix'] - time.time()
    if available < 1800:
        raise RuntimeError('Insufficient remaining allocation window to resume safely')
    cfg = TrainConfig(**json.loads((source / 'config.json').read_text()))
    cfg.epochs = math.ceil(target)
    cfg.max_hours = (baseline_seconds + available - 1200) / 3600
    cfg.export_reserve_seconds = 300
    cfg.checkpoint_every_steps = 10**9  # time-based durable checkpoints below
    cfg.resume = True
    cfg.patience = 16
    cfg.validate()
    report = {'status': 'preparing', 'phase': 'prepare', 'run_id': settings['run_id'],
        'initial_step': original['step'], 'initial_epoch_fraction': baseline_progress,
        'additional_epochs_requested': 5, 'target_epoch_fraction': target,
        'backup_interval_seconds': 1800, 'early_stopping_min_delta': .0005,
        'early_stopping_patience_validations': 16, 'early_stopping_min_additional_epochs': 1,
        'deadline_unix': settings['deadline_unix'], 'dtype': 'torch.bfloat16'}

    def publish():
        report['new_session_seconds'] = time.monotonic() - started
        atomic_json(reports / 'job_status.json', report)
        store.put(reports / 'job_status.json', 'reports/job_status.json')

    def backup(force=False):
        files = [p for p in source.rglob('*') if p.is_file() and not p.name.endswith('.tmp')
                 and not any(part.startswith('.staging-') for part in p.parts)
                 and not p.name.startswith(('step-', 'pre_smoke'))]
        # All complete artifacts precede the durable progress marker.
        for path in files:
            store.put(path, str(path.relative_to(work)))
        for path in reports.glob('*'):
            if path.is_file() and path.name != 'job_status.json':
                store.put(path, str(path.relative_to(work)))
        report['last_backup_unix'] = time.time()
        publish()

    class ContinueTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.signature == settings['signature'] and self.step == settings['initial_step']
            self.last_backup = time.monotonic()
            self.last_publish = 0.
            self.durations = []
            self.significant_best = self.best_score
            self.quality_stale = 0
            self.in_evaluation = False
            self.smoking = False
            self.target_reached = False

        def progress(self):
            return self.epoch + self.cursor / len(self.data.fit_rows)

        def update_report(self):
            report.update(step=self.step, epoch_fraction=self.progress(), full_epochs=self.epoch,
                additional_epochs_completed=self.progress()-baseline_progress,
                best_macro_ranking_ndcg20=self.best_score, quality_stale_validations=self.quality_stale,
                accumulated_training_seconds=self.elapsed())
            if self.durations:
                mean = float(np.mean(self.durations[-1000:]))
                report['mean_step_seconds'] = mean
                report['remaining_training_hours_estimate'] = max(0., target-self.progress()) * len(self.data.fit_rows) / self.effective_batch * mean / 3600

        def check_budget(self, reserve=False):
            super().check_budget(reserve)
            if reserve and not self.in_evaluation and not self.smoking:
                self.update_report()
                if self.progress() >= target - 1e-10:
                    self.target_reached = True
                    raise BudgetExpired('Requested five additional epochs completed')
                if time.monotonic() - self.last_backup >= 1800:
                    self.save()
                    backup()
                    self.last_backup = time.monotonic()
                if time.monotonic() - self.last_publish >= 60:
                    publish()
                    self.last_publish = time.monotonic()

        def optimizer_step(self, rows):
            torch.cuda.synchronize()
            before = time.perf_counter()
            loss = super().optimizer_step(rows)
            torch.cuda.synchronize()
            self.durations.append(time.perf_counter()-before)
            return loss

        @torch.inference_mode()
        def evaluate(self, rows):
            from evaluate_top200 import score_candidates, KS
            self.in_evaluation = True
            try:
                vectors = self.item_vectors()
                result = []
                for start in range(0, len(rows), self.config.evaluation_batch):
                    self.check_budget()
                    chosen = rows[start:start+self.config.evaluation_batch]
                    with self.autocast():
                        query = self.model.query_tower(**self.inputs(chosen)).float()
                    scores = query @ vectors.T
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('Nonfinite validation scores')
                    rankings = torch.argsort(scores, dim=1, descending=True, stable=True).cpu().numpy()
                    for row_no, ranking in zip(chosen, rankings):
                        row = self.data.pairs.iloc[row_no].to_dict()
                        row['selection_validation'] = True
                        ids = self.data.scorer.ids
                        class Ranking:
                            def __iter__(self):
                                return (ids[int(i)] for i in ranking)
                        scored = score_candidates(self.data.scorer, Ranking(), row,
                            self.data.listened[int(row['user_id'])])
                        scored.update(ranking=scored['ndcg@20'], discovery=scored['discovery_ndcg@20'],
                            diversity=np.nan, top20=scored['top200'][:20], new_top20=scored['discovery_top200'][:20])
                        result.append(scored)
                summary, frame, by_type = self.data.scorer.summarize(result)
                columns = [f'recall@{k}' for k in KS] + [f'discovery_recall@{k}' for k in KS]
                columns += ['ndcg@20','ndcg@100','ndcg@200','mrr@200','discovery_ndcg@20']
                averages = frame.groupby('query_type')[columns].mean()
                summary['retrieval_micro'] = {k: float(v) for k,v in frame[columns].mean().items()}
                summary['retrieval_macro'] = {k: float(v) for k,v in averages.mean().items()}
                return summary, frame, by_type
            finally:
                self.in_evaluation = False

        def validate(self):
            summary = super().validate()
            score = summary['macro_ranking_ndcg20']
            if score > self.significant_best + .0005:
                self.significant_best, self.quality_stale = score, 0
            else:
                self.quality_stale += 1
            report['latest_validation'] = summary
            report['latest_validation_step'] = self.step
            atomic_json(reports / 'validation_latest.json', {'step':self.step, **summary})
            store.put(reports / 'validation_latest.json', 'reports/validation_latest.json')
            self.update_report()
            publish()
            return summary

        def should_stop_early(self):
            return self.progress()-baseline_progress >= 1 and self.quality_stale >= 16

    def terminate(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, terminate)
    trainer = None
    try:
        torch.set_num_threads(4)
        device, dtype = select_device(cfg)
        assert 'RTX PRO 6000' in torch.cuda.get_device_name() and dtype == torch.bfloat16
        report['gpu'] = torch.cuda.get_device_name()
        budget = ExperimentBudget(artifacts, code/'music4all_crs', cfg, started_at=started)
        budget.previous = baseline_seconds
        tokenizer = _cached_pretrained(AutoTokenizer.from_pretrained, cfg.qwen_model,
            revision=cfg.qwen_revision, cache_dir=str(artifacts/'hf_cache'), trust_remote_code=False)
        data = prepare_data(code/'music4all_crs', artifacts, cfg, tokenizer, device, deadline=budget)
        assert data.fingerprint == settings['data_fingerprint'] and len(data.fit_rows) == 511897
        assert len(data.validation_rows) == 2600 and len(data.catalog) == 64016
        model, tokenizer = make_model(cfg, data, device, dtype, artifacts/'hf_cache', tokenizer=tokenizer)
        trainer = ContinueTrainer(model, data, cfg, artifacts, device, dtype, budget=budget, started_at=started)
        assert trainer.run_dir == source
        # Benchmark real resumed optimizer state, then restore all weights/RNG/cursor.
        report['phase'] = 'smoke'
        publish()
        trainer.save('pre_smoke.pt')
        trainer.smoking = True
        rng = np.random.default_rng(cfg.seed+913)
        for _ in range(110):
            trainer.check_budget(reserve=True)
            loss = trainer.optimizer_step(rng.choice(data.fit_rows, trainer.effective_batch, replace=False))
            if not math.isfinite(loss):
                raise FloatingPointError('Nonfinite resumed smoke loss')
        measured = trainer.durations[10:]
        trainer.restore(source/'pre_smoke.pt')
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.smoking = False
        assert trainer.step == settings['initial_step'] and abs(trainer.progress()-baseline_progress)<1e-12
        # Exact equality of restored trainable weights and optimizer tensors.
        restored = trainer._state()
        for tower in ('query','item'):
            assert all(torch.equal(restored[tower][k], original['weights'][tower][k]) for k in restored[tower])
        for key, moment in trainer.optimizer.state_dict()['state'].items():
            assert all(torch.equal(v.cpu(), original['optimizer']['state'][key][k]) if isinstance(v,torch.Tensor) else v==original['optimizer']['state'][key][k] for k,v in moment.items())
        before = time.monotonic()
        summary, _, _ = trainer.evaluate(data.validation_rows)
        validation_seconds = time.monotonic()-before
        report['smoke'] = {'passed':True, 'measured_steps':100, 'mean_step_seconds':float(np.mean(measured)),
            'validation_seconds':validation_seconds, 'weights_optimizer_rng_restored':True,
            'initial_step':trainer.step, 'five_additional_epochs_training_hours':5*len(data.fit_rows)/32*float(np.mean(measured))/3600}
        report['latest_validation'] = summary
        trainer.durations.clear()
        trainer.save()
        backup(force=True)
        trainer.last_backup = time.monotonic()
        del original, restored
        gc.collect()
        report.update(status='training',phase='training')
        publish()
        trainer.fit()
        reason = json.loads((source/'status.json').read_text())['reason']
        if trainer.target_reached:
            reason = 'five_additional_epochs_completed'
        report.update(training_reason=reason, phase='export')
        trainer.update_report()
        backup(force=True)
        bundle = export_bundle(trainer, tokenizer)
        best = torch.load(source/'best.pt',map_location='cpu',weights_only=True)
        report.update(bundle=str(bundle), best_checkpoint_step=best['step'],
            best_checkpoint_epoch_fraction=best['epoch']+best['cursor']/len(data.fit_rows),
            status='trained',phase='ready_for_full_evaluation')
        backup(force=True)
        atomic_json(root/'training_result.json',report)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        (reports/'traceback.txt').write_text(traceback.format_exc())
        store.put(reports/'traceback.txt','reports/traceback.txt')
        if trainer is not None:
            backup(force=True)
        raise
    finally:
        publish()

if __name__ == '__main__':
    main()
