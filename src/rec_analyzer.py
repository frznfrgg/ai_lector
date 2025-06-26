import json
import os
import uuid
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import librosa
import xtts_inference
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer
from voicefixer import VoiceFixer

STOPWORDS_PATH = "stopwords.txt"
MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"
ABSTRACT_PROMPT = "Ты - помощник-конспетолог. Дальше будет дан длинный текст лекции. Твоя задача - составить по нему подробный, информативный и связный конспект. Конспект должен быть понятен человеку, не читавшему лекцию, и давать хорошее понимание темы лекции"
EMOT_AN_PROMPT = "Ты - анализитор лекционных фрагментов. Дальше будет дан отрывок лекции. Оцени его по содержанию и эмоциональной подаче. Используй не больше 2 ярких прилагательных и ничего больше"
MINDMAP_PROMPT = 'Ты — интеллектуальный ассистент, который умеет превращать лекции в структурированные интеллект-карты (mindmap). Проанализируй текст лекции, которая будет дальше, и выдели из него основные темы, подтемы и детали. Структурируй информацию в иерархическую JSON-структуру, отражающую смысловую организацию материала. \
                Формат JSON: каждая тема — это ключ, а её значение либо массив подтем, либо объект с вложенными подтемами. Если нет дальнейших деталей, используй пустой массив. \
                Пример:\
                {"title": "название лекции", "nodes": [{"id": "название текста", "label": "Название текста", "children": [{"id": "название микротемы", "label": "Название микротемы", "children":[{"id": "название микротемы", "label": "Название микротемы", "children": []}]}]}]}'
QUESTIONS_PROMPT = "Ты - автор учебных пособий с большим опытом. По тексту лекции, который будет дан дальше, создай вопросы для самопроверки на знание материала.\
    Вопросы должны быть на разные темы и разные по сложности и должны охватывать ключевые факты, причино-следственные связи и важные детали"
PODCAST_PROMPT = "Ты — опытный редактор образовательных подкастов. Твоя задача — по тексту лекции и списку вопросов создать сценарий диалога между ведущим подкаста и автором лекции. \
    Ведущий задаёт вопросы, а лектор отвечает на них подробно, понятно и строго по содержанию лекции. \
    Диалог должен начинаться с короткого приветствия от ведущего и лектора, а завершаться тёплым прощанием от обоих. \
    Пиши исключительно на русском языке. Все даты, века, порядковые и количественные числительные обязательно пиши только словами (например: «девятнадцатый век», «две тысячи третий год», «тринадцать вариантов»). \
    Не используй формулы, переменные, математические обозначения, символы, LaTeX-нотацию или аббревиатуры (например, M(n), $2^{n-1}$, log, n). Переписывай смысл полностью словами. \
    Не используй заглушки вроде [имя преподавателя] — обращайся нейтрально, например: «наш гость», «автор лекции» и т.д. \
    Ответ верни строго в формате JSON, где каждая часть диалога — это отдельный элемент: \
    {\"part_1\": {\"presenter\": \"слова ведущего\", \"lector\": \"слова лектора\"}, \
    \"part_2\": {\"presenter\": \"вопрос ведущего\", \"lector\": \"ответ лектора\"}, ...} \
    Стиль речи — литературный, но живой и дружелюбный. Соблюдай структуру строго, не добавляй ничего вне JSON. Обязательно заверши диалог прощанием. Обязательно придерживайся всех сказанных правил, ни в коем случае не нарушай их"
CLEAN_PODCAST = "Перепиши данный JSON-диалог, строго соблюдая следующие требования: \
1. Пиши исключительно на русском языке. Не допускается использование слов, символов, переменных или выражений на английском (например, function, log, n-graph и т.п.). \
2. Заменяй все математические обозначения, переменные и формулы (например, M(n), n-1, n-однородный, $2^{n-1}$, log, f(x)) на полноценные пояснения словами, понятные широкой аудитории. Примеры: \
- 'n-однородный гиперграф' → 'гиперграф, в котором каждое ребро соединяет одинаковое количество вершин' \
- '$2^{n-1}$' → 'два в степени эн минус один' \
- 'M(n)'' → 'функция, значение которой зависит от количества элементов' \
3. Не оставляй никаких формул, переменных, символов или сокращений в тексте. Всё должно быть записано литературным, полным и понятным языком. \
4. Все числа, годы и века — пиши словами. Например: «две тысячи третий год», «девятнадцатый век», «шестнадцать вариантов». \
5. Удали любые плейсхолдеры (например, [имя преподавателя]) и замени их на «лектор», «автор лекции» или аналогичные формулировки. \
Верни только исправленный JSON-объект. Не добавляй никаких пояснений, комментариев или лишнего текста."

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
            "wav_path": self._prepair_audio,
            "waveform": self._prepair_audio,
            "sample_rate": self._prepair_audio,
            "abstract_text": self._yandexgpt_analyze,
            "questions": self._yandexgpt_analyze,
            "podcast_text": self._yandexgpt_analyze,
            "mind_map": self._yandexgpt_analyze,
            "final_chunks": self._yandexgpt_analyze,
            "fragment" : self._extract_clean_fragment,
            "clean_fragment": self._extract_clean_fragment,
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
    
    def _clean_json(self, text: str):
        """transform string to JSON"""
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    
    def _clear_gpu_cache(self):
        """Clear the GPU cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_stat(self):
        """Calculates statistics for diagram, and creates chunks labeled by speaker."""
        self._clear_gpu_cache()
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
        self._clear_gpu_cache()
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

        for speaker, start, end in self.labeled_chunks:
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            fragment = self.waveform[:, start_sample:end_sample]

            if fragment.shape[0] > 1:
                fragment = fragment.mean(dim=0, keepdim=True)

            fragment = fragment.squeeze(0)
            fragment_np = fragment.numpy()

            target_sample_rate = processor.feature_extractor.sampling_rate
            if self.sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, new_freq=target_sample_rate
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
            chunks.append([speaker, text.strip(), (start, end)])
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
        silence_intervals = [i[2][1]-i[2][0] for i in self.chunks if i[0]==3]
        mean = sum(silence_intervals) / len(silence_intervals)
        for i in range(len(_transcripted_chunks)):
            if _transcripted_chunks[i][0] == 3:
                if (
                    float(_transcripted_chunks[i][2][1])
                    - float(_transcripted_chunks[i][2][0])
                    <= mean
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

    def _yandexgpt_analyze(self):
        """Analyzes text using yandexgpt to generate abstract of text, questions, podcast text with answers, mind map and summarized."""
        self._clear_gpu_cache()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            torch_dtype="auto")
        
        abstract_message = [{"role": "user", "content": f"{ABSTRACT_PROMPT}: {self.lecture_text}"}]
        abstract_input_ids = tokenizer.apply_chat_template(
            abstract_message, tokenize=True, return_tensors="pt").to("cuda")
        outputs_abstract = model.generate(abstract_input_ids, max_new_tokens=32000)
        abstract = tokenizer.decode(outputs_abstract[0][abstract_input_ids.size(1) :], skip_special_tokens=True)
        self._cache["abstract_text"] = abstract

        final_chunks = deepcopy(self.transcripted_chunks)
        for i in range(len(final_chunks)):
            if final_chunks[i][0] == 1:
                emot_analyz_mess = [{"role": "user", "content": f"{EMOT_AN_PROMPT}: {final_chunks[i][1]}"}]
                emot_analyz_input_ids = tokenizer.apply_chat_template(
                    emot_analyz_mess, tokenize=True, return_tensors="pt"
                ).to("cuda")
                outputs_emot_analyz = model.generate(emot_analyz_input_ids, max_new_tokens=10)
                emot_analyz = tokenizer.decode(outputs_emot_analyz[0][emot_analyz_input_ids.size(1) :], skip_special_tokens=True)
                final_chunks[i].append(emot_analyz)
        self._cache["final_chunks"] = final_chunks

        mindmap_message = [{"role": "user", "content": f"{MINDMAP_PROMPT}: {self.lecture_text}"}]
        mindmap_input_ids = tokenizer.apply_chat_template(
            mindmap_message, tokenize=True, return_tensors="pt"
        ).to("cuda")
        mindmap_output = model.generate(mindmap_input_ids, max_new_tokens=32000)
        mindmap = tokenizer.decode(mindmap_output[0][mindmap_input_ids.size(1) :], skip_special_tokens=True)
        cleaned_mindmap = self._clean_json(mindmap)
        cleaned_mindmap = json.dumps(cleaned_mindmap, indent=4, ensure_ascii=False)
        self._cache["mind_map"] = cleaned_mindmap

        quest_message = [{"role": "user", "content": f"{QUESTIONS_PROMPT}: {self.lecture_text}"}]
        quest_input_ids = tokenizer.apply_chat_template(
            quest_message, tokenize=True, return_tensors="pt"
        ).to("cuda")
        quests_output = model.generate(quest_input_ids, max_new_tokens=32000)
        quests = tokenizer.decode(quests_output[0][quest_input_ids.size(1) :], skip_special_tokens=True)
        self._cache['questions'] = quests

        podcast_message = [{"role": "user", "content": f"{PODCAST_PROMPT}: Лекция: {self.lecture_text}. Вопросы: {quests}"}]
        podcast_input_ids = tokenizer.apply_chat_template(
            podcast_message, tokenize=True, return_tensors="pt"
        ).to("cuda")
        podcast_output = model.generate(podcast_input_ids, max_new_tokens=32000)
        podcast_text = tokenizer.decode(podcast_output[0][podcast_input_ids.size(1) :], skip_special_tokens=True)

        clean_podcast_prompt = [{"role": "user", "content": f"{CLEAN_PODCAST}: {self._clean_json(podcast_text)}"}]
        clean_podcast_input_ids = tokenizer.apply_chat_template(
            clean_podcast_prompt, tokenize=True, return_tensors="pt").to("cuda")
        outputs_clean_podcast = model.generate(clean_podcast_input_ids, max_new_tokens=32000)
        clean_podcast = tokenizer.decode(outputs_clean_podcast[0][clean_podcast_input_ids.size(1) :], skip_special_tokens=True)
        self._cache["podcast_text"] = self._clean_json(clean_podcast)
        self._clear_gpu_cache()
        
    def _prepair_audio(self):
        """Converts audio from mp3 to wav."""
        if ".mp3" in self.recording_path:
            waveform, sample_rate = torchaudio.load(self.recording_path)
            wav_path = self.recording_path[:self.recording_path.find(".")] + ".wav"
            torchaudio.save(wav_path, waveform, sample_rate)
            self._cache["wav_path"] = wav_path
            self._cache["waveform"] = waveform
            self._cache["sample_rate"] = sample_rate
        elif ".wav" in self.recording_path:
            waveform, sample_rate = torchaudio.load(self.recording_path)
            self._cache["wav_path"] = self.recording_path
            self._cache["waveform"] = waveform
            self._cache["sample_rate"] = sample_rate
    
    def _extract_clean_fragment(self, target_duration=60, sr=22050):
        """Extracting a fragment with the lectors clear voice.

        Args:
            target_duration (int, optional): length of clean fragment of lectors voice. Defaults to 60.
            sr (int, optional): sample rate of a  fragment. Defaults to 22050.
        """
        _waveform = self.waveform.mean(dim=0).numpy()  # mono

        if self.sample_rate != sr:
            _waveform = librosa.resample(y=_waveform, orig_sr=self.sample_rate, target_sr=sr)

        trimmed, _ = librosa.effects.trim(_waveform, top_db=30)

        total_len = len(trimmed)
        max_samples = target_duration * sr

        if total_len <= max_samples:
            fragment = trimmed
        else:
            start = (total_len - max_samples) // 2
            end = start + max_samples
            fragment = trimmed[start:end]
        self._cache["fragment"] = fragment
        torchaudio.save(f"clean_{self.wav_path}", torch.tensor(fragment).unsqueeze(0), sr)
        self._cache["clean_fragment"] = f"clean_{self.wav_path}"
        voicefixer = VoiceFixer()
        voicefixer.restore(input=self.clean_fragment, output=self.clean_fragment)

    def _generate_podcast(self):
        """Generates podcast using XTTS-v2."""

        podcast_text = deepcopy(self.podcast_text)
        
        xtts_model = xtts_inference.XttsInference()

        final_audio = np.array([])
        for i in podcast_text.keys():
            for speaker in podcast_text[i].keys():
                if "presenter" in speaker:
                    trans, presenter_speach = xtts_model(podcast_text[i]["presenter"], "../utils/podcast_host.wav")
                if "lector" in speaker:
                    trans, lector_speach = xtts_model(podcast_text[i]["lector"], self.clean_fragment)
            final_audio = np.concatenate((final_audio, presenter_speach[0], lector_speach[0]))

        output_file_path = str(uuid.uuid4())

        torchaudio.save(output_file_path + ".wav", torch.tensor(final_audio).unsqueeze(0), 24000)

        audio = AudioSegment.from_file(output_file_path + ".wav", format="wav")

        audio.export(output_file_path + ".mp3", format="mp3", bitrate="192k")

        self._cache["path_to_podcast"] = output_file_path + ".mp3"

        os.remove(output_file_path + ".wav")
        os.remove(self.wav_path)
        os.remove(self.clean_fragment)
        self._clear_gpu_cache()
