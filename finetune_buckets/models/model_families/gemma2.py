from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

default_system_prompt = ""


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model.config._name_or_path, add_eos_token=False, add_bos_token=False
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side
    
    return model, tokenizer


class Gemma2StringConverter:
    def string_formatter(example):
        """Parse OpenAI style chat format to Gemma 2 format for fine-tuning."""
        
        BOS, EOS = "<bos>", "<eos>"
        USER_START, MODEL_START = "<start_of_turn>user\n", "<start_of_turn>model\n"
        END = "<end_of_turn>\n"
        
        if "messages" not in example:
            raise ValueError("No messages in the example")
        
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt = 0
        if messages[0]["role"] != "system":
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]["content"]
            pt = 1
        
        # Gemma 2 doesn't have a dedicated system prompt block, incorporate into first user message
        str_message = BOS + USER_START
        if system_prompt:
            str_message += system_prompt + "\n"
        
        first_round = True
        
        if pt == len(messages):
            raise ValueError("The message should be user - assistant alternation")
        
        while pt < len(messages):
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")
            
            if first_round:
                str_message += messages[pt]["content"] + END
                first_round = False
            else:
                str_message += USER_START + messages[pt]["content"] + END
            
            pt += 1
            
            if pt >= len(messages):
                raise ValueError("The message should be user - assistant alternation")
            
            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += MODEL_START + messages[pt]["content"]
            pt += 1
            
            if pt == len(messages):
                str_message += " " + EOS
            else:
                str_message += END
        
        return {"text": str_message}
    
    def string_formatter_completion_only(example):
        """Parse OpenAI style chat format to Gemma 2 format for inference."""
        
        BOS = "<bos>"
        USER_START, MODEL_START = "<start_of_turn>user\n", "<start_of_turn>model\n"
        END = "<end_of_turn>\n"
        
        if "messages" not in example:
            raise ValueError("No messages in the example")
        
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt = 0
        if messages[0]["role"] != "system":
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]["content"]
            pt = 1
        
        str_message = BOS + USER_START
        if system_prompt:
            str_message += system_prompt + "\n"
        
        first_round = True
        
        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")
            
            if first_round:
                str_message += messages[pt]["content"] + END
                first_round = False
            else:
                str_message += USER_START + messages[pt]["content"] + END
            
            pt += 1
            if pt >= len(messages) - 1:
                break
            
            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += MODEL_START + messages[pt]["content"] + END
            pt += 1
        
        if messages[-1]["role"] != "assistant":
            raise ValueError("Completion only mode should end with a header of assistant message")
        
        str_message += MODEL_START + messages[-1]["content"]
        
        return {"text": str_message}
    
    def conversion_to_gemma2_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(
            Gemma2StringConverter.string_formatter, remove_columns=redundant_columns
        )
        return dataset


from transformers import StoppingCriteria, StoppingCriteriaList


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords):
        self.stops = keywords
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        input_ids = input_ids.cpu()
        for seq in input_ids:
            for stop in self.stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    return True
        return False


gemma2_stop_key_words = torch.LongTensor([[107]])  # <end_of_turn>
gemma2_stopping_criteria = StoppingCriteriaList(
    [KeywordStoppingCriteria(keywords=gemma2_stop_key_words)]
)
