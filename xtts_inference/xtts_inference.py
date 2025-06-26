import os
from huggingface_hub import snapshot_download
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from omogre import Transcriptor
from typing import List
import numpy as np

class XttsInference:
    def __init__(self, xtts_model_path: str = '../xtts_ru', transcriptor_data_path: str = '../omogre_data'):
        """
        Initialize the XTTS model and the transcriptor.

        Args:
            xtts_model_path (str): Path to the XTTS model directory.
            transcriptor_data_path (str): Path to transcriptor data.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xtts_model_path = xtts_model_path
        self.transcriptor = Transcriptor(data_path=transcriptor_data_path)

        self._download_model_if_needed()
        self._load_model()
    
    def _clear_gpu_cache(self):
        """Clear the GPU cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _download_model_if_needed(self):
        if not os.path.isfile("../xtts_ru/model.pth"):
            print("Model directory is missing or empty. Cloning from Hugging Face")
            snapshot_download("omogr/xtts-ru-ipa", local_dir=self.xtts_model_path, local_dir_use_symlinks=False, allow_patterns=["*.pth"])
            print("Model downloaded.")

    def _load_model(self):
        config_path = os.path.join(self.xtts_model_path, "config.json")
        checkpoint_path = os.path.join(self.xtts_model_path, "model.pth")
        vocab_path = os.path.join(self.xtts_model_path, "vocab.json")

        if not all(map(os.path.exists, [config_path, checkpoint_path, vocab_path])):
            raise FileNotFoundError("Missing model files in the specified path.")

        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)

        model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=False,
            speaker_file_path='-'
        )
        model.to(self.device)
        model.eval()

        self.model = model
        self.config = config

    def _split_text(self, text: str, max_length: int = 140) -> List:
        """
        Split text

        Args:
            text (str): text

        Returns:
            List: splitted text
        """
        splitted_text = []
        while len(text) > max_length:
            split_pos = max([text[:max_length].rfind(i) for i in ".!?"])
            if split_pos == -1:
                split_pos = max([text[:max_length].rfind(i) for i in ",:-)"])
                if split_pos == -1:
                    split_pos = text[:max_length].rfind(" ")
                    if split_pos == -1:
                        split_pos = max_length
            splitted_text.append(text[:(split_pos + 1)].strip())
            text = text[(split_pos + 1):].strip()
        if text:
            splitted_text.append(text)
        
        return splitted_text

    def _normalize_volume(self, audio, target_dbfs: float = -18.0):
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        current_dbfs = 20 * np.log10(rms + 1e-9)
        required_gain = 10 ** ((target_dbfs - current_dbfs) / 20)
        normalized_audio = audio * required_gain
        return np.clip(normalized_audio, -1.0, 1.0)

    def __call__(self, src_text: str, reference_audio: str) -> tuple[str, torch.Tensor]:
        """
        Generate synthesized speech from input text and reference speaker audio.

        Args:
            src_text (str): Text to be synthesized.xtts_inference
        Returns:
            tuple: (transcribed text, audio waveform tensor [1, T])
        """
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        gpt_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=reference_audio,
            gpt_cond_len=self.config.gpt_cond_len,
            max_ref_length=self.config.max_ref_len,
            sound_norm_refs=self.config.sound_norm_refs
        )

        transcripted_text = "".join(self.transcriptor([src_text]))
        splitted_text = self._split_text(transcripted_text)
        audio = np.array([])
        pause = np.zeros(2000, dtype=np.float32)

        for tts_text in splitted_text:
            out = self.model.inference(
                text=tts_text,
                language='ru',
                gpt_cond_latent=gpt_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.config.temperature,
                length_penalty=self.config.length_penalty,
                repetition_penalty=self.config.repetition_penalty,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            audio = np.concatenate((audio, pause, out["wav"]))

        audio = self._normalize_volume(audio)
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        self._clear_gpu_cache()
        return "".join(splitted_text), audio_tensor


