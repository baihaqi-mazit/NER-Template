import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
import spacy
from spacy import Language
from pathlib import Path

@spacy.registry.loggers("my_custom_logger.v1")
def custom_logger(log_path):
    def setup_logger(
        nlp: Language,
        stdout: IO=sys.stdout,
        stderr: IO=sys.stderr
    ) -> Tuple[Callable, Callable]:
        stdout.write(f"Logging to {log_path}\n")
        log_file = Path(log_path).open("w", encoding="utf8")
        log_file.write("step\t")
        log_file.write("score\t")
        for pipe in nlp.pipe_names:
            log_file.write(f"loss_{pipe}\t")
        log_file.write("\n")

        def log_step(info: Optional[Dict[str, Any]]):
            if info:
                log_file.write(f"{info['step']}\t")
                log_file.write(f"{info['score']}\t")
                for pipe in nlp.pipe_names:
                    log_file.write(f"{info['losses'][pipe]}\t")
                log_file.write("\n")

        def finalize():
            log_file.close()

        return log_step, finalize

    return setup_logger