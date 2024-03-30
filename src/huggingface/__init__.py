from enum import StrEnum
from huggingface_hub import HfApi


class HuggingFaceTask(StrEnum):
    AudioClassification = "audio-classification"
    AutomaticSpeechRecognition = "automatic-speech-recognition"
    Conversational = "conversational"
    DepthEstimation = "depth-estimation"
    DocumentQuestionAnswering = "document-question-answering"
    FeatureExtraction = "feature-extraction"
    FillMask = "fill-mask"
    ImageClassification = "image-classification"
    ImageFeatureExtraction = "image-feature-extraction"
    ImageSegmentation = "image-segmentation"
    ImageToImage = "image-to-image"
    ImageToText = "image-to-text"
    MaskGeneration = "mask-generation"
    ObjectDetection = "object-detection"
    QuestionAnswering = "question-answering"
    Summarization = "summarization"
    TableQuestionAnswering = "table-question-answering"
    Text2TextGeneration = "text2text-generation"
    TextClassification = "text-classification"
    TextGeneration = "text-generation"
    TextToAudio = "text-to-audio"
    TokenClassification = "token-classification"
    Translation = "translation"
    TranslationXXtoYY = "translation_xx_to_yy"
    VideoClassification = "video-classification"
    VisualQuestionAnswering = "visual-question-answering"
    ZeroShotClassification = "zero-shot-classification"
    ZeroShotImageClassification = "zero-shot-image-classification"
    ZeroShotAudioClassification = "zero-shot-audio-classification"
    ZeroShotObjectDetection = "zero-shot-object-detection"


AVAILABLE_PIPELINES = ["feature-extraction",
                       "text-classification", "token-classification", "question-answering",
                       "table-question-answering", "fill-mask", "summarization", "translation",
                       "text2text-generation", "text-generation", "zero-shot-classification", "conversational",
                       "image-classification", "translation_XX_to_YY"]

hf_api = HfApi()
