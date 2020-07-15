# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """

from collections import OrderedDict

from transformers.configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    GPT2Config,
    LongformerConfig,
    OpenAIGPTConfig,
    ReformerConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)
from transformers.configuration_marian import MarianConfig
from transformers.configuration_utils import PretrainedConfig

from modelings import (
    BertForSequenceScore,
    RobertaForSequenceScore,
)

MODEL_FOR_SEQUENCE_SCORE_MAPPING = OrderedDict(
    [
        (BertConfig, BertForSequenceScore),
        # (DistilBertConfig, DistilBertForSequenceScore),
        # (AlbertConfig, AlbertForSequenceScore),
        # (CamembertConfig, CamembertForSequenceScore),
        # (XLMRobertaConfig, XLMRobertaForSequenceScore),
        # (BartConfig, BartForSequenceScore),
        # (LongformerConfig, LongformerForSequenceScore),
        (RobertaConfig, RobertaForSequenceScore),
        # (BertConfig, BertForSequenceScore),
        # (XLNetConfig, XLNetForSequenceScore),
        # (FlaubertConfig, FlaubertForSequenceScore),
        # (XLMConfig, XLMForSequenceScore),
        # (ElectraConfig, ElectraForSequenceScore),
    ]
)


class AutoModelForSequenceScore:
    r"""
        :class:`~transformers.AutoModelForSequenceScore` is a generic model class
        that will be instantiated as one of the sequence Score model classes of the library
        when created with the `AutoModelForSequenceScore.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForSequenceScore is designed to be instantiated "
            "using the `AutoModelForSequenceScore.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceScore.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers.DistilBertForSequenceScore` (DistilBERT model)
                - isInstance of `albert` configuration class: :class:`~transformers.AlbertForSequenceScore` (ALBERT model)
                - isInstance of `camembert` configuration class: :class:`~transformers.CamembertForSequenceScore` (CamemBERT model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformers.XLMRobertaForSequenceScore` (XLM-RoBERTa model)
                - isInstance of `roberta` configuration class: :class:`~transformers.RobertaForSequenceScore` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformers.BertForSequenceScore` (Bert model)
                - isInstance of `xlnet` configuration class: :class:`~transformers.XLNetForSequenceScore` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformers.XLMForSequenceScore` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformers.FlaubertForSequenceScore` (Flaubert model)


        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceScore.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_SEQUENCE_SCORE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_SCORE_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence Score model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: :class:`~transformers.DistilBertForSequenceScore` (DistilBERT model)
            - `albert`: :class:`~transformers.AlbertForSequenceScore` (ALBERT model)
            - `camembert`: :class:`~transformers.CamembertForSequenceScore` (CamemBERT model)
            - `xlm-roberta`: :class:`~transformers.XLMRobertaForSequenceScore` (XLM-RoBERTa model)
            - `roberta`: :class:`~transformers.RobertaForSequenceScore` (RoBERTa model)
            - `bert`: :class:`~transformers.BertForSequenceScore` (Bert model)
            - `xlnet`: :class:`~transformers.XLNetForSequenceScore` (XLNet model)
            - `flaubert`: :class:`~transformers.FlaubertForSequenceScore` (Flaubert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForSequenceScore.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceScore.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceScore.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_SCORE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_SCORE_MAPPING.keys()),
            )
        )
