from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf


def finetune(train, valid, test, model_name='roberta-base'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(list(train), truncation=True, padding=True)
    val_encodings = tokenizer(list(valid), truncation=True, padding=True)
    test_encodings = tokenizer(list(test), truncation=True, padding=True)
    # todo: complete
    pass