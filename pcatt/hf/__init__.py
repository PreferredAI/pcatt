from pcatt.hf.greedtok import GreedTok
from transformers import PretrainedConfig, AutoConfig, AutoTokenizer

class GreedTokConfig(PretrainedConfig):
    model_type = "greedtok"

    def __init__():
        pass

AutoConfig.register("greedtok", GreedTokConfig)
AutoTokenizer.register(GreedTokConfig, GreedTok)
