from django.conf import settings
import os

TAG_RANGE = list(range(1,40))

MODEL_DIR = os.path.join(settings.BASE_DIR, 'datasets')
MODEL_NAME = MODEL_DIR + 'pretrained_multiclass_model'

TOKENIZER_DIR = ''
TOKENIZER = TOKENIZER_DIR + "DeepPavlov/rubert-base-cased"


STAISTICS_DOC_NAME = MODEL_DIR + 'statistics_document.json'

