import torch


def load_models():
    """Load only Silero VAD. Whisper is handled by STTEngine."""
    silero_vad, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    return None, silero_vad