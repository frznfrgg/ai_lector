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

        #       список из пар вида ["спикер", время речи]
        self.diogram = None
        #       сырой текст лекции
        self.lecture_text = None
        #       текст конспекта
        self.abstract_text = None
        #       доля стоп слов в процентах
        self.stop_words_rate = None
        #       список с кол-вом слов в секунду
        self.words_per_second = None
        #       список с кол-вом слов сказанных суммарно к моменту времени
        self.words_counter = None
        #       среднее число слов в секунду
        self.avg_words_speed = None
        #       самое большое число слов с секунду
        self.max_words_speed = None
        #       момент времени с наибольшим числом слов в сек
        self.time_of_top_speed = None
        #       строка в которой лежат вопросы к лекции
        self.questions = None
        #       список с фрагметами текста по временным отрезкам
        self.chunks = None
        #       список из пар 10 самых популярных слов и их частоты
        self.top_words = None
        #       список с тройками вида ("спикер", начало, конец)
        self.labeled_chunks = None
        #       моменты времени с известными значениями кол-ва слов (выступают в роли абсциссы для графиков)
        self.seconds = None

    def get_lecture_text(self) -> str:
        if self.lecture_text is None:
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

        else:
            pass

        return self.lecture_text

    def get_abstract_text(self) -> str:
        if self.abstract_text is None:
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
        else:
            pass

        return self.abstract_text

    def get_questions(self) -> str:
        if self.questions is None:
            _ = self.get_abstract_text()
        else:
            pass
        return self.questions

    def get_top_words(self) -> List[Tuple[int, str]]:
        if self.top_words is None:
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

        else:
            pass

        return self.top_words

    def get_diogram(self) -> List:
        if self.diogram is None:
            self._get_stat()
        else:
            pass

        return self.diogram

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

    def _get_stat(self):
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

    def get_labeled_chunks(self) -> List[Tuple[str, float, float]]:
        if self.labeled_chunks is None:
            self._get_stat()
        else:
            pass

        return self.labeled_chunks

    def get_stop_words_rate(self) -> int:
        if self.stop_words_rate is None:
            with open(STOP_WORDS_PATH) as f:
                stop_words = set(f.read().splitlines())

            preproc_text_list = [
                word for word in self.lecture_text.split() if word in stop_words
            ]

            self.stop_words_rate = int(
                len(preproc_text_list) / len(self.lecture_text.split()) * 100.0
            )
        else:
            pass

        return self.stop_words_rate

    #   возвращает пару списков Х и У для построения графика слов сказанных за все время
    def get_words_counter(self) -> Tuple[List[int]]:
        if self.words_counter is None or self.seconds is None:
            seconds = [0]
            words_counter = [0]
            previous_end = 0

            if self.chunks is None:
                _ = self.get_lecture_text()
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
        else:
            pass

        return self.seconds, self.words_counter

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

    def get_words_per_second(self) -> Tuple[List]:
        if self.words_per_second is None:
            if self.seconds is None or self.words_counter is None:
                _, _ = self.get_words_counter()
            else:
                pass

            self.words_per_second = self._calculate_derivative().tolist()
        else:
            pass

        return self.seconds, self.words_per_second

    def get_avg_words_speed(self) -> float:
        if self.avg_words_speed is None:
            if self.words_per_second is None:
                _, _ = self.get_words_per_second()
            else:
                pass

            self.avg_words_speed = float(np.mean(self.words_per_second))
        else:
            pass

        return self.avg_words_speed

    def get_max_words_speed(self) -> float:
        if self.max_words_speed is None:
            if self.words_per_second is None:
                _, _ = self.get_words_per_second()
            else:
                pass

            self.max_words_speed = float(np.max(self.words_per_second))
        else:
            pass

        return self.max_words_speed

    def get_time_of_top_speed(self) -> float:
        if self.time_of_top_speed is None:
            if self.words_per_second is None:
                _, _ = self.get_words_per_second()
            else:
                pass

            self.time_of_top_speed = float(
                self.seconds[np.argmax(self.words_per_second)]
            )
        else:
            pass

        return self.time_of_top_speed
