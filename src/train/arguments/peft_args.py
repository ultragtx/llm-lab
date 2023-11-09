from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PeftArguments:
    lora_rank: int = field(
        default=None,
        metadata={"help": "LoRA rank number"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha weight"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to LoRA checkpoints"}
    )
    
    pre_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "Prefix encoder length for P-Tuning V2"}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether add projection layers for prefix encoder"}
    )
    ptuning_checkpoint: str = field(
        default=None, 
        metadata={"help": "Path to P-Tuning V2 checkpoints"}
    )