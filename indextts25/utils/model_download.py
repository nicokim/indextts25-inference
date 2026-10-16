"""Auxiliary models of IndexTTS-2.5: w2v-bert-2.0, CAMPPlus and BigVGAN.

They are used from `{model_dir}/hf_cache/` if already present (that is how the IndexTTS
distribution leaves them); otherwise only the needed files are downloaded from the Hub
into its standard cache. For a mirror, the HF_ENDPOINT variable is enough.
"""
import os


def ensure_models_available(model_dir: str) -> dict:
    """Returns the local paths {'w2v_bert': dir, 'campplus': file, 'bigvgan': dir}."""
    local = os.path.join(model_dir, "hf_cache")
    w2v = os.path.join(local, "w2v-bert-2.0")
    campplus = os.path.join(local, "campplus_cn_common.bin")
    vocoder = os.path.join(local, "bigvgan")

    # Check the files, not the directories: an interrupted download leaves the
    # directory created, and it was never retried.
    if not os.path.isfile(os.path.join(w2v, "model.safetensors")):
        from huggingface_hub import snapshot_download
        w2v = snapshot_download("facebook/w2v-bert-2.0", allow_patterns=[
            "config.json", "preprocessor_config.json", "model.safetensors"])
    if not os.path.isfile(campplus):
        from huggingface_hub import hf_hub_download
        campplus = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
    if not os.path.isfile(os.path.join(vocoder, "bigvgan_generator.pt")):
        from huggingface_hub import snapshot_download
        vocoder = snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x",
                                    allow_patterns=["config.json", "bigvgan_generator.pt"])
    return {"w2v_bert": w2v, "campplus": campplus, "bigvgan": vocoder}
