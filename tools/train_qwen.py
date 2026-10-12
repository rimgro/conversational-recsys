"""Portable training / total-epoch resume using the signed trainer core."""
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time

os.environ['USE_TF'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/qwen_g4.json')
    p.add_argument('--epochs', type=int, help='Total epochs, including previous training')
    p.add_argument('--max-hours', type=float, help='Cumulative active-time limit, including previous invocations')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    from transformers import AutoTokenizer
    from recsys.two_tower_qwen.config import TrainConfig
    from recsys.two_tower_qwen.data import ExperimentBudget, prepare_data
    from recsys.two_tower_qwen.training import Trainer, make_model, select_device
    from recsys.two_tower_qwen.model import _cached_pretrained
    from recsys.two_tower_qwen.bundle import export_bundle
    started = time.monotonic()
    cfg = TrainConfig(**json.loads(args.config.read_text()))
    cfg.resume = args.resume
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_hours is not None:
        cfg.max_hours = args.max_hours
    cfg.validate()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    device, dtype = select_device(cfg)
    budget = ExperimentBudget(args.artifacts, args.data_dir, cfg, started_at=started)
    tokenizer = _cached_pretrained(AutoTokenizer.from_pretrained, cfg.qwen_model,
        revision=cfg.qwen_revision, cache_dir=str(args.artifacts / 'hf_cache'), trust_remote_code=False)
    data = prepare_data(args.data_dir, args.artifacts, cfg, tokenizer, device, deadline=budget)
    model, tokenizer = make_model(cfg, data, device, dtype, args.artifacts / 'hf_cache', tokenizer=tokenizer)
    trainer = Trainer(model, data, cfg, args.artifacts, device, dtype, budget=budget, started_at=started)
    if args.resume and not (trainer.run_dir / 'last.pt').is_file():
        raise FileNotFoundError('No matching last.pt; refusing to resume from step zero')
    budget.previous = max(budget.previous, trainer.elapsed_previous)
    budget.record()
    def terminate(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, terminate)
    trainer.fit()
    bundle = export_bundle(trainer, tokenizer)
    print(json.dumps({'run_dir': str(trainer.run_dir), 'bundle': str(bundle),
        'training_status': json.loads((trainer.run_dir / 'status.json').read_text())}, ensure_ascii=False))

if __name__ == '__main__':
    main()
