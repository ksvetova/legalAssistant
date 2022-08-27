import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from IPython.display import clear_output
import numpy as np
import ast
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import docx2txt
import re

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class TextVectorizerBERT:
    ''' Класс векторизации текстовых данных.'''

    # Загружаем предобученныею модель токенизации текста
    def __init__(self, tokenizer_path, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)

    # Приводим текст в его векторное представление
    def vectorize(self, text):
        encoded_input = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=24,
            return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings[0].tolist()


class ClassificationModel:
    '''
      Модель классификации векторных представлений текста.
      Общий объём тегов классификации: 39
    '''

    dtset_col_txt_name = 'content'
    dtset_col_tag_name = 'tag'
    dtset_col_vec_name = 'vec_content'

    # Инициализируем векторизатор и классификатор
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

        self.model = CatBoostClassifier(
            iterations=30,
            loss_function='MultiClass',
            learning_rate=0.1,
            depth=8,
            eval_metric='TotalF1:average=Macro')

    # Загружаем датасет для обучения модели-классификатора
    def load_dataset(self, path):
        self.dataset = pd.read_csv(path, sep=',')
        print(f"Drop nan. shape before {self.dataset.shape}.")
        self.dataset = self.dataset.dropna()
        print(f"Drop nan. shape after {self.dataset.shape}")

        self.dataset[self.dtset_col_tag_name] = self.dataset[self.dtset_col_tag_name].astype('int32')

        if self.dtset_col_vec_name in self.dataset.columns:
            self.dataset[self.dtset_col_vec_name] = self.dataset[self.dtset_col_vec_name].apply(
                lambda vec: ast.literal_eval(vec))

        counter = Counter(self.dataset[model.dtset_col_tag_name])
        print("Частота классов:")
        for k, v in counter.items():
            print(f"{k}: {v}")

    # Получение векторных представлений текста
    def vectorize_dataset(self):
        if self.dtset_col_vec_name not in self.dataset.columns:
            self.vectorized_content = list()
            for i, text in enumerate(self.dataset[self.dtset_col_txt_name]):
                clear_output(wait=True)
                print(f"{i}/{self.dataset.shape[0]}")
                self.vectorized_content.append(self.vectorizer.vectorize(text))

            print(f"Сохраняем векторные представления в {DATASET_NAME}")
            self.dataset[self.dtset_col_vec_name] = self.vectorized_content
            self.dataset.to_csv(DATASET_NAME, sep=',')
        else:
            self.vectorized_content = self.dataset[self.dtset_col_vec_name]

    # Разбиение датасета на тренировочную и тестовую выборки
    def split_dataset(self, d_split=0.4):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.vectorized_content,
            self.dataset[self.dtset_col_tag_name],
            test_size=d_split,
            random_state=0,
            stratify=self.dataset[self.dtset_col_tag_name])

    # Обучение модели-классификатора и сохранение её в файл
    def train_model(self, model_name):
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_val, self.y_val),
            plot=True)

        self.model.save_model(model_name)

    # Загрузка предобученной модели классификатора
    def load_model(self, model_path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    # Получение предсказания для заданного векторного представления текста.
    # Формат: (предсказанный класс, точность предсказания)
    def predict_tag(self, vector):
        prediction = self.model.predict_proba(vector)
        highest_score = max(zip(TAG_RANGE, prediction), key=lambda pair: pair[1])
        return highest_score


def parse_doc(doc_path):
    raw_text = docx2txt.process(doc_path)

    # Удаляем шапку документа
    ignorecase_head_ptr = re.compile(
        '\n{2,}утверждены\n{2,}постановлением правительства\n{2,}Российской Федерации\n{2,}', re.IGNORECASE)
    pos = ignorecase_head_ptr.search(raw_text).span()[1]
    raw_text = raw_text[pos:]

    print('here1')

    # Удаляем дополнительные Приложения в хвосте документа
    tail_ptr1 = re.compile('\n{3,}приложение( \w+){,2}\n{2}', re.IGNORECASE)
    ptr = tail_ptr1.search(raw_text)
    if ptr:
        pos = ptr.span()[0]
        raw_text = raw_text[:pos]

    print('here2')

    # Удаляем новые заголовки с текстом в хвосте документа
    tail_ptr2 = re.compile('(([А-Я,"]+[ ]*)+\n{2,}){3,}')
    title_span = tail_ptr2.search(raw_text).span()

    print('here3')

    ptr = tail_ptr2.search(raw_text[title_span[1]:])
    if ptr:
        pos = ptr.span()[0]
        raw_text = raw_text[pos + title_span[1]:]
    raw_text = raw_text[title_span[1]:]

    # Удаляем мусор из анализируемой части документа
    parts = list(filter(lambda v: v and 'www.consultant.ru' not in v
                                  and 'Список изменяющих документов\n\n(в ред. Постановлений Правительства РФ' not in v
                                  and v not in [' ', '  '],
                        re.split('\n{5,}', raw_text)))

    raw_text = '\n\n'.join(map(lambda v: v.strip('\n'), parts))
    raw_text = re.sub('\n{4} \n{4}', ' ', raw_text)

    # Разбиваем текст на абзацы
    parts = list(filter(lambda v: v != ' ' and v, re.split('\n\n', raw_text)))

    # Разбиваем абзацы на предложения
    sents = []
    for part in parts:
        sents += sent_tokenize(part)
    sents = list(filter(lambda v: len(v) > 5, sents))

    return sents


def create_statistics_json(predictions):
    pass


def analyze_doc():
    print('HERE!')
    pass