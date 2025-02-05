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
QUESTIONS_PROMPT = "Приведи 4 вопроса для самопроверки по материалу этого же текста"
TREE_PROMPT = 'По этому же тексту создай подробное дерево знаний в формате json. Вот пример того, что у тебя должно получитьтся: \nconst jsonData = {\n"title":"Лекция по математическому анализу",\n"nodes":[\n{\n"id":"матанализ", "label":"Математический анализ", "children":[...]\n}\n]\n};. Даже если дерево получится маленьким, все равно создай его. В ответ отправь только json, не пиши ничего больще.'


class LectureHelper:
    """Audio recording analyzer class.
    NOTE: All attributes that correspond to metrics are stored in _cache, which is used to provide lazy initialization functionality.

    Attributes:
        _cache (dict): Stores calculated metrics
    Attributes stored in _cache:
        lecture_text (str): Full text of lection
        abstract_text (str): Summarized text of lection
        diagram (List[Tuple[str, float]]): Statistics for pie chart representing active time for each speaker
        stopwords_rate (int): Percentage of stopwords
        words_per_second (List[float]): Statistics graph showing how fast the words were spoken
        words_counter (List[int]): Statistics of how many words were spoken by each timestamp
        avg_words_speed (float): Average speed of speech
        max_words_speed (float): Maximal speed of speech
        time_of_top_speed (float): Time (in seconds) when speech was the fastest
        questions (str): Generated questions for lection
        chunks (List[dict]]): Full text of lection splitted in chunks. Each item in list consists of a timestamp and a text
        top_words (List[Tuple[int, str]]): List of the most popular words and number of their occasions
        labeled_chunks (List[list]): Lecture text splitted into chunks by speaker. Each item consists of speaker label, start time of chunk, end time of chunk
        seconds (List[float]): List of timestamps in seconds. Used as an abcissa in graphs
    """

    def __init__(
        self, recording_path: str, gigachat_api_key: str, pyannote_api_key: str
    ):
        """Initializes an analyzer object.

        Args:
            recording_path (str): path to file with the necessary audio file
            gigachat_api_key (str): secret api key for accessing GigaChat api service
            pyannote_api_key (str): secret api key for accessing pyannote model from Huggingface

        Raises:
            FileNotFoundError: raised if path to the file could not be found
        """
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
            "abstract_text": self._set_abstract_text,
            "diagram": self._set_stat,
            "stopwords_rate": self._set_stopwords_rate,
            "words_per_second": self._set_words_per_second,
            "words_counter": self._set_words_counter,
            "questions": self._set_abstract_text,
            "chunks": self._set_lecture_text,
            "labeled_chunks": self._set_stat,
            "seconds": self._set_words_counter,
            "mind_map": self._set_abstract_text,
            "transcripted_chunks": self._set_transcripted_chunks,
            "speed": self._set_speech_speed,
            "popular_words": self._set_popular_words,
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
            default=str
        )

    def _set_lecture_text(self):
        """Creates transcription of the recording and text of the lection splitted into chunks."""
        lecture_text = ""
        for _, text, _ in self.transcripted_chunks:
            lecture_text += text

        self._cache["lecture_text"] = lecture_text

    def _set_abstract_text(self):
        """Creates summarized text of the lection and questions for lection."""
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

            # запрос на создание 4 вопросов
            payload.messages.append(
                Messages(role=MessagesRole.USER, content=QUESTIONS_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["questions"] = response.choices[0].message.content

            # запрос на дерево знаний
            payload.messages.append(
                Messages(role=MessagesRole.USER, content=TREE_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self._cache["mind_map"] = response.choices[0].message.content

    def _set_popular_words(self):
        """Calculates the most common words."""
        with open(STOPWORDS_PATH) as f:
            stopwords = set(f.read().splitlines())
        words = {"lector": [], "audience": []}

        for speaker, text, _ in self.transcripted_chunks:
            speaker = speaker.lower()
            if speaker != "silence":
                words[speaker].extend(
                    [
                        word
                        for word in text.lower().split()
                        if word not in stopwords and word.isalpha()
                    ]
                )
        word_counts_lector = Counter(words["lector"])
        word_counts_audience = Counter(words["audience"])
        popular_words = [
            word_counts_lector.most_common()[:10],
            word_counts_audience.most_common()[:10],
        ]

        self._cache["popular_words"] = popular_words

    def _fill_silence_intervals(
        self,
        data: List[Tuple[str, float, float]],
    ) -> List[Tuple[str, float, float]]:
        filled_data = []

        if data[0][1] > 0:
            filled_data.append(["Silence", 0, data[0][1]])
        for i, entry in enumerate(data):
            speaker, start, end = entry
            filled_data.append(entry)

            if i < len(data) - 1:
                next_start = data[i + 1][1]
                if end < next_start:
                    filled_data.append(["Silence", end, next_start])
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
                timestamps_of_speakers[i][0] = "Lector"
            else:
                timestamps_of_speakers[i][0] = "Audience"

        time_of_events = t_lecturer + t_audience + t_silence

        self._cache["diagram"] = {
            "lecturer": t_lecturer / time_of_events * 100.0,
            "discussion": t_audience / time_of_events * 100.0,
            "quiet": t_silence / time_of_events * 100.0,
        }
        self._cache["labeled_chunks"] = self._fill_silence_intervals(
            timestamps_of_speakers
        )

    def _set_stopwords_rate(self):
        """Calculates percentage of stopwords."""
        with open(STOPWORDS_PATH) as f:
            stopwords = set(f.read().splitlines())

        preproc_text_list = [
            word
            for word in self.lecture_text.split()
            if word in stopwords
        ]

        self._cache["stopwords_rate"] = int(
            len(preproc_text_list)
            / len(self.lecture_text.split())
            * 100.0
        )

    def _set_words_counter(self):
        """Calculates a total amount of words spoken by the certain time and timestamps in seconds."""
        word_counter = [0]
        seconds = [0]
        for _, text, timestamps in self.transcripted_chunks:
            start, end = timestamps
            word_counter.append(word_counter[-1] + len(text.split()))
            seconds.append(end)
        self._cache["words_counter"] = word_counter[1:]
        self._cache["seconds"] = seconds[1:]

    def _gaussian_smoothing(self, array: np.ndarray, degree=5) -> np.array:
        """Applies gaussian smoothing of chosen degree to the array.
        Used in calculating derivative method.

        Args:
            array (np.ndarray): The array to smooth.
            degree (int, optional): The degree (strength) of smoothing. Defaults to 5.

        Returns:
            np.array: Smoothed array.
        """
        myarray = np.pad(array, (degree - 1, degree - 1), mode="edge")
        window = degree * 2 - 1

        weight = np.arange(-degree + 1, degree) / window
        weight = np.exp(-(16 * weight**2))
        weight /= weight.sum()

        smoothed = np.convolve(myarray, weight, mode="valid")
        return smoothed

    def _calculate_derivative(
        self,
        x,
        y,
        max_limit=7,
        smoothing="gaussian",
        smooth_degree=20,
    ) -> np.ndarray:
        """Calculates the derivative of y with respect of x.

        Args:
            x (int): Sequence of tics on x axis
            y (int): Sequence of values on y axis
            max_limit (int, optional): If provided, imits the max value of the derivative. Defaults to 7
            smoothing (str, optional): If provided, specifies the type of smoothing to use. Defaults to "gaussian"
            smooth_degree (int, optional): Sets degree of smoothing. Defaults to 20

        Returns:
            np.ndarray: the values of calculated derivative
        """

        derivative = np.gradient(y, x)
        if max_limit:
            derivative[derivative >= max_limit] = max_limit
        derivative = np.nan_to_num(derivative, nan=np.nanmean(derivative))

        if smoothing == "gaussian":
            derivative = self._gaussian_smoothing(derivative, degree=smooth_degree)

        return derivative

    def _set_words_per_second(self):
        """Calculates speed of speech at each timestamp."""
        words_per_second = {}
        for words, seconds in zip(self.words_counter, self.seconds):
            words_per_second[words] = seconds
        
        self._cache["words_per_second"] = self._calculate_derivative(
            sorted(list(words_per_second.values())), sorted(list(words_per_second.keys()))
        ).tolist()

    def _set_speech_speed(self):
        speed = dict(zip(self.seconds, self.words_per_second))
        self._cache["speed"] = speed

    def _set_transcripted_chunks(self):
        transcripted_chunks = []

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

            if speaker != "Silence":
                text = speech_recognition_pipe(
                    inputs=fragment_np,
                    generate_kwargs={"language": "russian"},
                    return_timestamps=True,
                )["text"]
            else:
                text = ""

            if (
                speaker == "Silence"
                or text.strip() == ""
                or text == " Продолжение следует..."
            ):
                text = ""
                speaker = "Silence"
            else:
                text = speech_recognition_pipe(
                    inputs=fragment_np,
                    generate_kwargs={"language": "russian"},
                    return_timestamps=True,
                )["text"]

            transcripted_chunks.append([speaker, text, (start, end)])
        self._cache["transcripted_chunks"] = transcripted_chunks
