import json
import os
import uuid
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pyannote.audio import Pipeline
from pydub import AudioSegment
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# store api keys as env variable and access them like in example below:
# PYANNOTE_AUTH_TOKEN = os.environ.get("PYANNOTE_API_KEY")
# GIGACHAT_API_KEY = os.environ.get("GIGACHAT_API_KEY")

STOPWORDS_PATH = "stopwords.txt"
ROLE_PROMPT = "Ты выступаешь в роли автора учебо-методических пособий для высшего учебного заведения"
ABSTRACT_PROMPT = "Сделай конспект по тексту"
QUESTIONS_PROMPT = "Приведи вопросы для самопроверки по материалу этого же текста"
ANSWERS_PROMT = "Напиши текст подкаста, в котором ведущий задает эти вопросы, а лектор по материалу лекции на них отвечает. \
    Придерживайся следующих правил: не добавляй ничего от себя и не делай вступление и завершение, \
    но на вопросы отвечай очень развернуто, перед каждой репликой пиши ее автора - ведущий или лектор, не разделяй никак одну реплику, никак не выделяй слова в твоем ответе, \
    используй двоеточие только когда пишешь автора речи и речь должна быть от первого лица."

TREE_PROMPT = 'Создай по этому же тексту подробное дерево знаний. Придерживайся следующих правил: \
    В дереве ость только одна главная тема, которая содержит другие микротемы. \
    Дерево знаний должно быть глубоким, содержать много микротем. \
    Описание каждой темы должны состоять из словосочетаний или очень коротких предложений \
    У каждой темы обязательно должны быть поля id, topic и children. \
    Результат верни в формате JSON-массива без каких-либо пояснений, например: \
    {"id": "название текста", "topic": "Название текста", "children": [{"id": "название микротемы", "topic": "Название микротемы", "children":[{"id": "название микротемы", "topic": "Название микротемы", "children": []}]}]}.'


class LectureHelper:
    """Audio recording analyzer class.
    NOTE: All attributes that correspond to metrics are stored in _cache, which is used to provide lazy initialization functionality.

    Attributes:
        _cache (dict): Stores calculated metrics
    Attributes stored in _cache:
        lecture_text (str): Full text of lection
        abstract_text (str): Summarized text of lection
        questions (str): Generated questions for lection
        answers (str): Generated podcast text with answers om questions
        mind_map (str): JSON-like mindmap of lecture
        popular_words_no_stopw (List[Dict[str, int]]): List of the most popular words and number of their occasions without stopwords
        popular_words_w_stopw (List[Dict[str, int]]): List of the most popular words and number of their occasions with stopwords
        diagram (List[Tuple[str, float]]): Statistics for pie chart representing active time for each speaker
        syllables_per_minute (List[float]): Speed of speach in syllables/min
        speed (Dict[int, int]): Speed of speech at each minute
        chunks (List[dict]]): Full text of lection splitted in chunks. Each item in list consists of a speaker id, text and timestamp
        transcripted_chunks (List[list]): Chunks in readable format
        final chunks (List[list]): Chunks in readable format with emotional analysis
        wav_path (str): Path to wav audio of lection
        path_to_podcast (str): Path to wav audio of podcast
        labeled_chunks (List[List]): time allocation of speakers
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
            recorID (str): ID of audio in database

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
            "diagram": self._set_stat,
            "labeled_chunks": self._set_stat,
            "chunks": self._set_chunks,
            "lecture_text": self._set_lecture_text,
            "popular_words_no_stopw": self._set_popular_words,
            "popular_words_w_stopw": self._set_popular_words,
            "syllables_per_minute": self._set_syllables_per_minute,
            "speed": self._set_speech_speed,
            "transcripted_chunks": self._set_transcripted_chunks,
            "abstract_text": self._gigachat_analyze,
            "questions": self._gigachat_analyze,
            "answers": self._gigachat_analyze,
            "mind_map": self._gigachat_analyze,
            "final_chunks": self._gigachat_analyze,
            "wav_path": self._prepair_audio,
            "path_to_podcast": self._generate_podcast,
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
        """Json format of some attributes."""

        return json.dumps(
            {
                "lecture_text": self.lecture_text,
                "abstract_text": self.abstract_text,
                "speech_speed": self.speed,
                "mindmap": self.mind_map,
                "popular_words_no_stopw": self.popular_words_no_stopw,
                "popular_words_w_stopw": self.popular_words_w_stopw,
                "conversation_static": self.diagram,
                "lecture_timeline": self.final_chunks,
                "questions": self.questions,
                "podcast": self.path_to_podcast,
            },
            default=str,
        )

    def _fill_silence_intervals(
        self,
        data: List[Tuple[str, float, float]],
    ) -> List[Tuple[str, float, float]]:
        """Fills intervals when no words were spoken."""
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

        diarization = pipeline(file=self.wav_path)

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

    def _set_chunks(self):
        """Creates chunks in the folowing format: [speaker_id, text, (time_of_start, time_of_end)]."""
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
        waveform, orig_sample_rate = torchaudio.load(self.wav_path)

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

    def _set_lecture_text(self):
        """Creates transcription of the recording and text of the lection splitted into chunks."""
        lecture_text = ""
        for _, text, _ in self.chunks:
            lecture_text += text

        self._cache["lecture_text"] = lecture_text

    def _set_popular_words(self):
        """Calculates the most common words."""
        with open(STOPWORDS_PATH) as f:
            stopwords = set(f.read().splitlines())
        word_no_stopw = {1: [], 2: []}

        for speaker, text, _ in self.chunks:
            # if not silence
            if speaker != 3:
                word_no_stopw[speaker].extend(
                    [
                        word
                        for word in text.lower().split()
                        if word not in stopwords and word.isalpha()
                    ]
                )
        word_counts_lector_no_stopw = Counter(word_no_stopw[1])
        word_counts_audience_no_stopw = Counter(word_no_stopw[2])
        popular_words_no_stopw = [
            dict(word_counts_audience_no_stopw.most_common()[:10]),
            dict(word_counts_lector_no_stopw.most_common()[:10]),
        ]

        self._cache["popular_words_no_stopw"] = popular_words_no_stopw

        word_w_stopw = {1: [], 2: []}
        for speaker, text, _ in self.chunks:
            # if not silence
            if speaker != 3:
                word_w_stopw[speaker].extend(
                    [word for word in text.lower().split() if word.isalpha()]
                )
        word_counts_lector_w_stopw = Counter(word_w_stopw[1])
        word_counts_audience_w_stpow = Counter(word_w_stopw[2])
        popular_words_w_stopw = [
            dict(word_counts_audience_w_stpow.most_common()[:10]),
            dict(word_counts_lector_w_stopw.most_common()[:10]),
        ]

        self._cache["popular_words_w_stopw"] = popular_words_w_stopw

    def _set_syllables_per_minute(self):
        """Calculates speed of speech in syllables per minute."""
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
        """Calculates speed of speech at each minute."""
        seconds = [0]
        for _, _, timestamps in self.chunks:
            start, end = timestamps
            seconds.append(end)
        minutes = sorted(list(set([second // 60 for second in seconds])))
        speed = dict(zip(minutes, self.syllables_per_minute))
        self._cache["speed"] = speed

    def _set_transcripted_chunks(self):
        """Creates transcripted chunks in readable format."""
        _transcripted_chunks = deepcopy(self.chunks)
        del_ind = []

        for i in range(len(_transcripted_chunks)):
            if _transcripted_chunks[i][0] == 3:
                if (
                    int(_transcripted_chunks[i][2][1])
                    - int(_transcripted_chunks[i][2][0])
                    <= 4
                ):
                    del_ind.append(i)

        for i in del_ind[::-1]:
            del _transcripted_chunks[i]
            if _transcripted_chunks[i - 1][0] == _transcripted_chunks[i][0]:
                _transcripted_chunks[i - 1][1] += _transcripted_chunks[i][1]
                _transcripted_chunks[i - 1][2] = list(_transcripted_chunks[i - 1][2])
                _transcripted_chunks[i - 1][2][1] = _transcripted_chunks[i][2][1]
                del _transcripted_chunks[i]

        for i in range(len(_transcripted_chunks)):
            start, end = _transcripted_chunks[i][2]
            _transcripted_chunks[i][2] = f"{int(start // 60)}:{int(start % 60)}"
        self._cache["transcripted_chunks"] = _transcripted_chunks

    def _gigachat_analyze(self):
        """Analyzes text using gigachat to generate abstract of text, questions, podcast text with answers, mind map and summarized."""
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
            credentials=self.gigachat_api_key,
            verify_ssl_certs=False,
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
                Messages(role=MessagesRole.USER, content=TREE_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            mindmap = response.choices[0].message.content
            mindmap = json.loads(mindmap)
            mindmap = json.dumps(mindmap, indent=4, ensure_ascii=False)
            self._cache["mind_map"] = mindmap

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=QUESTIONS_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["questions"] = response.choices[0].message.content

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=ANSWERS_PROMT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["answers"] = response.choices[0].message.content

            transc_chunks_f_giga = deepcopy(self.transcripted_chunks)
            for i in range(len(transc_chunks_f_giga)):
                if transc_chunks_f_giga[i][0] == 1:
                    payload.messages.append(
                        Messages(
                            role=MessagesRole.USER,
                            content=f"Оцени настроение этого куска лекции{transc_chunks_f_giga[i][1]}\
                                        Придерживайся следующий правил: \
                                        Результ верни в виде строки, содержащей словосочетание или короткое предложение, описывающее лекцию. \
                                        Используй разные эпитеты чтобы точнее передать атмосферу на лекции \
                                        Например: 'Интересно и полезно' или 'увлекательно и сложно' или 'скучно и непонятно'.",
                        )
                    )
                    response = giga.chat(payload)
                    payload.messages.append(response.choices[0].message)
                    transc_chunks_f_giga[i].append(response.choices[0].message.content)
            self._cache["final_chunks"] = transc_chunks_f_giga

    def _prepair_audio(self):
        """Converts audio from mp3 to wav."""
        if ".mp3" in self.recording_path:
            audio = AudioSegment.from_mp3(self.recording_path)

            wav_file = self.recording_path[: self.recording_path.find(".")] + ".wav"
            audio.export(wav_file, format="wav")

            self._cache["wav_path"] = wav_file

    def _generate_podcast(self):
        """Generates podcast using XTTS-v2."""

        answ = self.answers
        podcast_text = answ.split("\n\n")

        for i in range(len(podcast_text)):
            if "**" in podcast_text[i]:
                podcast_text[i] = podcast_text[i].replace("**", "")

        result = []
        for i in podcast_text:
            result.append([i[: i.find(":")], i[i.find(":") + 2 :]])

        podcast_chunks = []
        max_length = 150
        for _, text in result:
            sublist = []
            while len(text) > max_length:
                split_pos = text[:max_length].rfind(".")
                if split_pos == -1:
                    split_pos = text[:max_length].rfind(" ")
                    if split_pos == -1:
                        split_pos = max_length
                sublist.append(text[: split_pos + 1].strip())
                text = text[split_pos + 1 :].strip()
            if text:
                sublist.append(text)

            podcast_chunks.append([_, sublist])

        config = XttsConfig()
        config.load_json("../XTTS-v2/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="../XTTS-v2/")
        model.cuda()

        fin_aud = np.array([])
        pause = np.zeros(5000, dtype=np.float32)
        for speech in podcast_chunks:
            host_aud = np.array([])
            if "Ведущий" in speech[0]:
                for i in speech[1]:
                    outputs_host = model.synthesize(
                        i,
                        config,
                        speaker_wav="../utils/podcast_host.wav",
                        gpt_cond_len=5,
                        language="ru",
                    )
                    host_aud = np.concatenate((host_aud, pause, outputs_host["wav"]))

            lector_aud = np.array([])
            if "Лектор" in speech[0]:
                for j in speech[1]:
                    outputs_lector = model.synthesize(
                        j,
                        config,
                        speaker_wav=self.wav_path,
                        gpt_cond_len=5,
                        language="ru",
                    )
                    lector_aud = np.concatenate(
                        (lector_aud, pause, outputs_lector["wav"])
                    )

            fin_aud = np.concatenate((fin_aud, host_aud, lector_aud))

        output_file_path = str(uuid.uuid4())

        write(output_file_path + ".wav", 24000, fin_aud)

        audio = AudioSegment.from_file(output_file_path + ".wav", format="wav")

        audio.export(output_file_path + ".mp3", format="mp3", bitrate="192k")

        self._cache["path_to_podcast"] = output_file_path + ".mp3"

        os.remove(output_file_path + ".wav")
        os.remove(self.wav_path)
