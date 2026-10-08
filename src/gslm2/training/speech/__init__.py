"""Speech leaf trainers: SSL encoder pretraining (:mod:`encoder`) and speech-unit LM (:mod:`lm`).

``SLMTrainer`` lives in :mod:`gslm2.training.speech.lm` and is imported directly rather than
re-exported here: it pulls in the model factory + unit extraction, so keeping it out of this
package ``__init__`` leaves ``import gslm2.training`` lightweight.
"""

from gslm2.training.speech.encoder import SpeechSSLTrainer

__all__ = ["SpeechSSLTrainer"]
