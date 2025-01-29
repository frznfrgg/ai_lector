import os
from typing import List, Tuple

import numpy as np
import torch
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

STOP_WORDS_PATH = "stop_words.txt"
PYANNOTE_AUTH_TOKEN = ""
ABSTRACT_PROMPT = "Сделай конспект по тексту"
QUESTIONS_PROMPT = "Приведи 4 вопроса для самопроверки по материалу этого же текста"


class LectureHelper:
    def __init__(self, audio_path: str, api_key_path: str = "api.txt"):
        self.api_key = None
        if os.path.exists(api_key_path):
            with open(api_key_path) as f:
                self.api_key = f.readline()
        else:
            print(f"Warning! api_key file was not found in {api_key_path}")

        self.audio_path = None
        if os.path.exists(audio_path):
            self.audio_path = audio_path
        else:
            print(f"Warning! audio_path {audio_path} does not exist")

        # stores attributes with already assigned values
        self._cache = {}

    def __getattr__(self, name):
        computations = {
            "diagram": self._set_stat,
            "lecture_text": self._set_lecture_text,
            "abstract_text": self._set_abstract_text,
            "stop_words_rate": self._set_stop_words_rate,
            "words_per_second": self._set_words_per_second,
            "words_counter": self._set_words_counter,
            "avg_words_speed": self._set_avg_words_speed,
            "max_words_speed": self._set_max_words_speed,
            "time_of_top_speed": self._set_time_of_top_speed,
            "questions": self._set_abstract_text,
            "chunks": self._set_lecture_text,
            "top_words": self._set_top_words,
            "labeled_chunks": self._set_stat,
            "seconds": self._set_words_counter,
        }

        if name in computations:
            if name not in self._cache:  # Compute and store only if not already set
                self._cache[name] = computations[name]()
            return self._cache[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def set_lecture_text(self) -> str:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        result = pipe(
            inputs=self.audio_path,
            generate_kwargs={"language": "russian"},
            return_timestamps=True,
        )

        self.lecture_text = result["text"]
        self.chunks = result["chunks"]

    def set_abstract_text(self) -> str:
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content="Ты выступаешь в роли автора учебо-методических пособий для высшего учебного заведения",
                )
            ],
            temperature=0.3,
        )
        with GigaChat(credentials=self.api_key, verify_ssl_certs=False) as giga:
            payload.messages.append(
                Messages(
                    role=MessagesRole.USER,
                    content=f"{ABSTRACT_PROMPT}: [{self.lecture_text}]",
                )
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self.abstract_text = response.choices[0].message.content

            payload.messages.append(
                Messages(role=MessagesRole.USER, content=QUESTIONS_PROMPT)
            )
            response = giga.chat(payload)
            payload.messages.append(response.choices[0].message)
            self.questions = response.choices[0].message.content

    def set_top_words(self) -> List[Tuple[int, str]]:
        lst_no = [".", ",", ":", "!", '"', "'", "[", "]", "-", "—", "(", ")"]
        lst = []

        for word in self.lecture_text.lower().split():
            if word not in lst_no:
                _word = word
                if word[-1] in lst_no:
                    _word = _word[:-1]
                if word[0] in lst_no:
                    _word = _word[1:]
                lst.append(_word)

        _dict = dict()
        for word in lst:
            _dict[word] = _dict.get(word, 0) + 1

        # сортируем словарь посредством формирования списка (значение, ключ)
        _list = []
        for key, value in _dict.items():
            _list.append((value, key))
            _list.sort(reverse=True)

        self.top_words = _list[0:10]

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

    def set_stat(self):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=PYANNOTE_AUTH_TOKEN,
        ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        diarization = pipeline(file=self.audio_path)

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

        self.diogram = [
            ["Lecturer", t_lecturer],
            ["Audience", t_audience],
            ["Silence", t_silence],
        ]
        self.labeled_chunks = self._fill_silence_intervals(timestamps_of_speakers)

    def set_stop_words_rate(self) -> int:
        with open(STOP_WORDS_PATH) as f:
            stop_words = set(f.read().splitlines())

        preproc_text_list = [
            word for word in self.lecture_text.split() if word in stop_words
        ]

        self.stop_words_rate = int(
            len(preproc_text_list) / len(self.lecture_text.split()) * 100.0
        )

    #   возвращает пару списков Х и У для построения графика слов сказанных за все время
    def set_words_counter(self) -> Tuple[List[int]]:
        seconds = [0]
        words_counter = [0]
        previous_end = 0

        if self.chunks is None:
            _ = self.set_lecture_text()
        else:
            pass

        for line in self.chunks:
            start, end = line["timestamp"]
            text = line["text"]
            if end >= previous_end:
                seconds.append(seconds[-1] + end - previous_end)
            else:
                seconds.append(seconds[-1] + 30 - previous_end)
            words_counter.append(words_counter[-1] + len(text.split()))
            previous_end = end

        self.words_counter = words_counter
        self.seconds = seconds

    def _gaussian_smoothing(self, array: np.ndarray, degree=5) -> np.array:
        """Applies gaussian smoothing of chosen degree to the array.

        Args:
            array (np.ndarray): The array to smooth.
            degree (int, optional): The degree (strength) of smoothing. Defaults to 5.

        Returns:
            np.array: Smoothed array
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
        max_limit=7,
        smoothing="gaussian",
        smooth_degree=20,
    ) -> np.ndarray:
        """Calculates the derivative of y with respect of x.

        Args:
            max_limit (int, optional): If provided, imits the max value of the derivative. Defaults to 7.
            smoothing (str, optional): If provided, specifies the type of smoothing to use. Defaults to "gaussian".
            smooth_degree (int, optional): Sets degree of smoothing. Defaults to 20.

        Returns:
            np.ndarray: the values of calculated derivative
        """

        derivative = np.gradient(self.words_counter, self.seconds)
        if max_limit:
            derivative[derivative >= max_limit] = max_limit
        derivative = np.nan_to_num(derivative, nan=np.nanmean(derivative))

        if smoothing == "gaussian":
            derivative = self._gaussian_smoothing(derivative, degree=smooth_degree)

        return derivative

    def set_words_per_second(self) -> Tuple[List]:
        self.words_per_second = self._calculate_derivative().tolist()

    def set_avg_words_speed(self) -> float:
        self.avg_words_speed = float(np.mean(self.words_per_second))

    def set_max_words_speed(self) -> float:
        self.max_words_speed = float(np.max(self.words_per_second))

    def set_time_of_top_speed(self) -> float:
        self.time_of_top_speed = float(self.seconds[np.argmax(self.words_per_second)])
