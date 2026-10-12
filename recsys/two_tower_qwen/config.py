from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    qwen_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    text_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    history_events: int = 256
    query_tokens: int = 128
    hash_buckets: int = 8192
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = .05
    embedding_dim: int = 256
    text_tokens: int = 192
    encoding_batch: int = 64
    lyric_words: int = 128
    lyric_overlap: int = 32
    lyric_max_chunks: int = 8
    max_categories: int = 48
    item_token_dim: int = 128
    item_layers: int = 2
    item_heads: int = 4
    item_feedforward: int = 512
    dropout: float = .1
    negatives: int = 1024
    temperature: float = .08
    micro_batch: int = 4
    accumulation: int = 8
    lora_lr: float = 1e-4
    new_weights_lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.
    epochs: int = 3
    validation_user_fraction: float = .1
    validation_per_type: int = 200
    evaluation_batch: int = 8
    item_evaluation_batch: int = 512
    validate_every_steps: int = 1000
    checkpoint_every_steps: int = 100
    patience: int = 10  # validation calls, not epochs
    early_stopping_min_epochs: int = 1
    keep_validation_checkpoints: int = 3
    gradient_checkpointing: bool = True
    max_hours: float = 24.
    export_reserve_seconds: int = 900
    require_cuda: bool = True
    min_free_vram_gib: float = 12.
    resume: bool = True
    max_users: int | None = None
    max_queries_per_user: int | None = None

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for name in ("history_events", "query_tokens", "hash_buckets", "lora_rank",
                     "embedding_dim", "negatives", "micro_batch", "accumulation",
                     "epochs", "validation_per_type", "validate_every_steps",
                     "checkpoint_every_steps", "evaluation_batch", "item_evaluation_batch", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("early_stopping_min_epochs", "keep_validation_checkpoints"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.validation_user_fraction < 1 or self.temperature <= 0:
            raise ValueError("Invalid split fraction or temperature")
        if self.max_hours <= 0 or not 0 <= self.export_reserve_seconds < self.max_hours * 3600:
            raise ValueError("Invalid wall time budget")

    def identity(self):
        """Operational limits may change on resume; model/data/training semantics may not."""
        result = self.to_dict()
        for name in ("resume", "require_cuda", "min_free_vram_gib", "max_hours",
                     "export_reserve_seconds", "epochs", "patience", "early_stopping_min_epochs",
                     "keep_validation_checkpoints",
                     "checkpoint_every_steps", "validate_every_steps"):
            result.pop(name)
        return result
