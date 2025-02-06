import json
import os
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# store api keys as env variable and access them like in example below:
# PYANNOTE_AUTH_TOKEN = os.environ.get("PYANNOTE_API_KEY")
# GIGACHAT_API_KEY = os.environ.get("GIGACHAT_API_KEY")

STOPWORDS_PATH = "stopwords.txt"
ROLE_PROMPT = "Ты выступаешь в роли автора учебо-методических пособий для высшего учебного заведения"
ABSTRACT_PROMPT = "Сделай конспект по тексту"
QUESTIONS_PROMPT = "Приведи 7 вопросов для самопроверки по материалу этого же текста"
TREE_PROMPT = 'Создай по этому же тексту подробное дерево знаний. Придерживайся следующих правил: \
    В дереве ость только одна главная тема, которая содержит другие микротемы. \
    Дерево знаний должно быть глубоким, содержать много микротем. \
    Описание каждой темы должны состоять из словосочетаний или очень коротких предложений \
    У каждой темы обязательно должны быть поля id, topic и children. \
    Результат верни в формате JSON-массива без каких-либо пояснений, например: \
    {"id": "название текста", "topic": "Название текста", "children": [{"id": "название микротемы", "topic": "Название микротемы", "children":[{"id": "название микротемы", "topic": "Название микротемы", "children": []}]}]}.'
MOOD_PROMPT = "По этому же тексту оцени общее настроение \
    Придерживайся следующий правил: \
    Результ верни в виде строки, содержащей словосочетание или короткое предложение, описывающее лекцию. \
    Используй разные эпитеты чтобы точнее передать атмосферу на лекции \
    Например: 'Интересно и полезно' или 'увлекательно и сложно' или 'скучно и непонятно'."


class LectureHelper:
    """Audio recording analyzer class.
    NOTE: All attributes that correspond to metrics are stored in _cache, which is used to provide lazy initialization functionality.

    Attributes:
        _cache (dict): Stores calculated metrics
    Attributes stored in _cache:
        lecture_text (str): Full text of lection
        abstract_text (str): Summarized text of lection
        questions (str): Generated questions for lection
        mind_map (str): JSON-like mindmap of lecture
        mood (str): Overall mood of the lecture
        popular_words (List[Dict[str, int]]): List of the most popular words and number of their occasions
        diagram (List[Tuple[str, float]]): Statistics for pie chart representing active time for each speaker
        syllables_per_minute (List[float]): Speed of speach in syllables/min
        speed (Dict[int, int]): Speed of speech at each minute
        chunks (List[dict]]): Full text of lection splitted in chunks. Each item in list consists of a speaker id, text and timestamp
        transcripted_chunks (List[list]): Chunks in readable format
    """

    def __init__(
        self,
        recording_path: str,
        gigachat_api_key: str,
        pyannote_api_key: str,
        recordId: str,
    ):
        """Initializes an analyzer object.

        Args:
            recording_path (str): path to file with the necessary audio file
            gigachat_api_key (str): secret api key for accessing GigaChat api service
            pyannote_api_key (str): secret api key for accessing pyannote model from Huggingface

        Raises:
            FileNotFoundError: raised if path to the file could not be found
        """
        self.recordId = recordId
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.gigachat_api_key = gigachat_api_key
        self.pyannote_api_key = pyannote_api_key
        if os.path.exists(recording_path):
            self.recording_path = recording_path
        else:
            raise FileNotFoundError(f"Audio_path {recording_path} does not exist")

        # stores attributes with already assigned values
        self._cache = {}

        self.computations = {
            "lecture_text": self._set_lecture_text,
            "abstract_text": self._gigachat_analyze,
            "questions": self._gigachat_analyze,
            "mind_map": self._gigachat_analyze,
            "mood": self._gigachat_analyze,
            "popular_words": self._set_popular_words,
            "diagram": self._set_stat,
            "syllables_per_minute": self._set_syllables_per_minute,
            "chunks": self._set_chunks,
            "labeled_chunks": self._set_stat,
            "transcripted_chunks": self._set_transcripted_chunks,
            "speed": self._set_speech_speed,
        }

    def __getattr__(self, name: str):
        """Method that is raised when the attribute is called.
        Used to provide lazy initialization functionality:
        metrics are calculated only when the atribute is called for the first time.

        Args:
            name (str): name of an attribute to reach

        Raises:
            AttributeError: raised only if the attribute doesn't exist (metric is not specified)

        Returns:
            Metric corresponding to the attribute
        """

        if name in self.computations:
            if name not in self._cache:  # Compute and store only if not already set
                self.computations[name]()
            return self._cache[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def get_results(self):
        """Json format of some attributes"""

        return json.dumps(
            {
                "lecture_text": self.lecture_text,
                "abstract_text": self.abstract_text,
                "speech_speed": self.speed,
                "mindmap": self.mind_map,
                "popular_words": self.popular_words,
                "conversation_static": self.diagram,
                "lecture_timeline": self.transcripted_chunks,
                "questions": self.questions,
            },
            default=str,
        )

    def _set_lecture_text(self):
        """Creates transcription of the recording and text of the lection splitted into chunks."""
        lecture_text = ""
        for _, text, _ in self.chunks:
            lecture_text += text

        self._cache["lecture_text"] = lecture_text

    def _gigachat_analyze(self):
        """Analyzes text using gigachat to generate abstract of text, questions, mind map and summarized lecture mood."""
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=ROLE_PROMPT,
                )
            ],
            temperature=0.3,
        )
        with GigaChat(
            credentials=self.gigachat_api_key, verify_ssl_certs=False
        ) as giga:
            payload.messages.append(
                Messages(
                    role=MessagesRole.USER,
                    content=f"{ABSTRACT_PROMPT}: [{self.lecture_text}]",
                )
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["abstract_text"] = response.choices[0].message.content

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=QUESTIONS_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["questions"] = response.choices[0].message.content

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=TREE_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            mindmap = response.choices[0].message.content
            mindmap = json.loads(mindmap)
            mindmap = json.dumps(mindmap, indent=4, ensure_ascii=False)
            self._cache["mind_map"] = mindmap

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=MOOD_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["mood"] = response.choices[0].message.content

    def _set_popular_words(self):
        """Calculates the most common words."""
        with open(STOPWORDS_PATH) as f:
            stopwords = set(f.read().splitlines())
        words = {1: [], 2: []}

        for speaker, text, _ in self.chunks:
            # if not silence
            if speaker != 3:
                words[speaker].extend(
                    [
                        word
                        for word in text.lower().split()
                        if word not in stopwords and word.isalpha()
                    ]
                )
        word_counts_lector = Counter(words[1])
        word_counts_audience = Counter(words[2])
        popular_words = [
            dict(word_counts_audience.most_common()[:10]),
            dict(word_counts_lector.most_common()[:10]),
        ]

        self._cache["popular_words"] = popular_words

    def _fill_silence_intervals(
        self,
        data: List[Tuple[str, float, float]],
    ) -> List[Tuple[str, float, float]]:
        """Fills intervals when no words were spoken"""
        filled_data = []

        if data[0][1] > 0:
            filled_data.append([3, 0, data[0][1]])
        for i, entry in enumerate(data):
            speaker, start, end = entry
            filled_data.append(entry)

            if i < len(data) - 1:
                next_start = data[i + 1][1]
                if end < next_start:
                    filled_data.append([3, end, next_start])
        return filled_data

    def _set_stat(self):
        """Calculates statistics for diagram, and creates chunks labeled by speaker."""
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.pyannote_api_key,
        ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        diarization = pipeline(file=self.recording_path)

        time_allocation = diarization.chart()
        t_lecturer = time_allocation[0][1]
        t_audience = sum(
            [time_allocation[i][1] for i in range(1, len(time_allocation))]
        )
        t_silence = (
            max([segment.end for segment in diarization.itersegments()])
            - t_lecturer
            - t_audience
        )

        timestamps_of_speakers = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            timestamps_of_speakers.append(
                [speaker, round(segment.start, 2), round(segment.end, 2)]
            )

        lector_id = time_allocation[0][0]
        for i in range(len(timestamps_of_speakers)):
            if timestamps_of_speakers[i][0] == lector_id:
                timestamps_of_speakers[i][0] = 1
            else:
                timestamps_of_speakers[i][0] = 2

        time_of_events = t_lecturer + t_audience + t_silence

        self._cache["diagram"] = {
            "lecturer": t_lecturer / time_of_events * 100.0,
            "discussion": t_audience / time_of_events * 100.0,
            "quiet": t_silence / time_of_events * 100.0,
        }
        self._cache["labeled_chunks"] = self._fill_silence_intervals(
            timestamps_of_speakers
        )

    def _set_syllables_per_minute(self):
        """Calculates speed of speech in syllables per minute"""
        vowels = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
        total_syllables = 0
        syllables_per_minute = {}
        for speaker, text, timestamp in self.chunks:
            if speaker != 3:
                _, end = timestamp
                end = end // 60
                if end not in syllables_per_minute.keys():
                    syllables_per_minute[end] = 0
                total_syllables += sum(text.count(vowel) for vowel in vowels)
                syllables_per_minute[end] = total_syllables
        self._cache["syllables_per_minute"] = np.gradient(
            list(syllables_per_minute.values()), list(syllables_per_minute.keys())
        ).tolist()

    def _set_speech_speed(self):
        """Calculates speed of speech at each minute"""
        seconds = [0]
        for _, _, timestamps in self.chunks:
            start, end = timestamps
            seconds.append(end)
        minutes = sorted(list(set([second // 60 for second in seconds])))
        speed = dict(zip(minutes, self.syllables_per_minute))
        self._cache["speed"] = speed

    def _set_chunks(self):
        """Creates chunks in the folowing format: [speaker_id, text, (time_of_start, time_of_end)]"""
        chunks = []

        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        speech_recognition_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        waveform, orig_sample_rate = torchaudio.load(self.recording_path)

        for speaker, start, end in self.labeled_chunks:
            start_sample = int(start * orig_sample_rate)
            end_sample = int(end * orig_sample_rate)
            fragment = waveform[:, start_sample:end_sample]

            # convert to mono if necessary (whisper expects mono audio)
            if fragment.shape[0] > 1:
                fragment = fragment.mean(dim=0, keepdim=True)

            # remove channel dimension (now shape: [1, samples] -> [samples])
            fragment = fragment.squeeze(0)
            fragment_np = fragment.numpy()

            # ensure the sampling rate matches what the feature extractor expects
            target_sample_rate = processor.feature_extractor.sampling_rate
            if orig_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sample_rate, new_freq=target_sample_rate
                )
                fragment_resampled = resampler(fragment.unsqueeze(0))
                fragment_resampled = fragment_resampled.squeeze(0)
                fragment_np = fragment_resampled.numpy()

            if speaker != 3:
                text = speech_recognition_pipe(
                    inputs=fragment_np,
                    generate_kwargs={"language": "russian"},
                    return_timestamps=True,
                )["text"]
            if speaker == 3 or text.strip() == "" or text == " Продолжение следует...":
                text = ""
                speaker = 3
            chunks.append([speaker, text, (start, end)])
        self._cache["chunks"] = chunks

    def _set_transcripted_chunks(self):
        """Creates transcripted chunks in readable format"""
        transcripted_chunks = []
        for speaker, text, timestamp in self.chunks:
            start, end = timestamp
            transcripted_chunks.append(
                [speaker, text, f"{int(start // 60)}:{int(start % 60)}"]
            )
        self._cache["transcripted_chunks"] = transcripted_chunks
