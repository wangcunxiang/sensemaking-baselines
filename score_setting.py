# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SCORE utils (dataset loading and evaluation) """


import logging
import os
from enum import Enum
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer
from transformers import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import pearson_and_spearman

logger = logging.getLogger(__name__)


class ScoreProcessor(DataProcessor):
    """Processor for the Score dataset."""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['statement'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            text_a = line[1]
            label = line[2]
            assert isinstance(text_a, str)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("Start loading test data")
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[1]
            label = line[2]
            assert isinstance(text_a, str)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return [None]


score_processors = {
    "score": ScoreProcessor,
}

score_output_modes = {
    "score": "regression",
}

score_tasks_num_labels = {
    "score": 1,
}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def score_compute_metrics(task_name, preds, labels): #
    assert len(preds) == len(labels)
    if task_name == "score":

        return {"pearson_and_spearman": pearson_and_spearman(preds, labels),
                "acc":simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)



def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )



def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("text_a: %s" % (example.text_a))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"
