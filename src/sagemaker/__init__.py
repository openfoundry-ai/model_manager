from enum import StrEnum


class EC2Instance(StrEnum):
    SMALL = "ml.m5.xlarge"
    MEDIUM = "ml.p3.2xlarge"
    LARGE = "ml.g5.12xlarge"


class SagemakerTask(StrEnum):
    AutomaticSpeechRecognition = "asr"
    AudioEmbedding = "audioembedding"
    Classification = "classification"
    Depth2img = "depth2img"
    ExtractiveQuestionAnswering = "eqa"
    FillMask = "fillmask"
    ImageClassification = "ic"
    ImageEmbedding = "icembedding"
    ImageGeneration = "imagegeneration"
    Inpainting = "inpainting"
    ImageSegmentation = "is"
    LLM = "llm"
    NamedEntityRecognition = "ner"
    ObjectDetection = "od"
    Od1 = "od1"
    Regression = "regression"
    SemanticSegmentation = "semseg"
    SentenceSimilarity = "sentencesimilarity"
    SentencePairClassification = "spc"
    Summarization = "summarization"
    TabTransformerClassification = "tabtransformerclassification"
    TabTransformerRegression = "tabtransformerregression"
    TextClassification = "tc"
    TcEmbedding = "tcembedding"
    Text2text = "text2text"
    TextEmbedding = "textembedding"
    TextGeneration = "textgeneration"
    TextGeneration1 = "textgeneration1"
    TextGeneration2 = "textgeneration2"
    TextGenerationJP = "textgenerationjp"
    TextGenerationNeuron = "textgenerationneuron"
    Translation = "translation"
    Txt2img = "txt2img"
    Upscaling = "upscaling"
    ZeroShotTextClassification = "zstc"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, SagemakerTask))
