from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

default_system_prompt = "Below is an instruction that describes a task, which starts with '### Instruction:'. Write a response that appropriately completes the request. This response is put after '### Response:'.\n\n"


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    tokenizer.padding_side = padding_side
    
    return model, tokenizer


class QwenStringConverter:
    def string_formatter(example):
        """Parse OpenAI style chat format to base Qwen2 format for fine-tuning."""
        
        ENDOFTEXT = "<|endoftext|>"
        USER_START = "\n### Instruction:\n"
        MODEL_START = "\n### Response:\n"
        END = "\n"
        
        if 'messages' not in example:
            raise ValueError("No messages in the example")
        
        messages = example['messages']
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1
        
        str_message = system_prompt
        first_round = True
        
        if pt == len(messages):
            raise ValueError("The message should be user - assistant alternation")
        
        while pt < len(messages):
            if messages[pt]['role'] != 'user':
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += USER_START + messages[pt]['content'] + END
            pt += 1
            
            if pt >= len(messages):
                raise ValueError("The message should be user - assistant alternation")
            
            if messages[pt]['role'] != 'assistant':
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += MODEL_START + messages[pt]['content']
            pt += 1
            
            if pt == len(messages):
                str_message += ENDOFTEXT
            else:
                str_message += END
        
        return {'text': str_message}
    
    def string_formatter_completion_only(example):
        """Parse OpenAI style chat format to base Qwen2 format for inference."""
        
        USER_START = "\n### Instruction:\n"
        MODEL_START = "\n### Response:\n"
        END = "\n"
        
        if 'messages' not in example:
            raise ValueError("No messages in the example")
        
        messages = example['messages']
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1
        
        str_message = system_prompt
        
        while pt < len(messages) - 1:
            if messages[pt]['role'] != 'user':
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += USER_START + messages[pt]['content'] + END
            pt += 1
            
            if pt >= len(messages) - 1:
                break
            
            if messages[pt]['role'] != 'assistant':
                raise ValueError("The message should be user - assistant alternation")
            
            str_message += MODEL_START + messages[pt]['content'] + END
            pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("Completion only mode should end with a header of assistant message")
        
        str_message += MODEL_START + messages[-1]['content']
        
        return {'text': str_message}
    
    def conversion_to_qwen_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(QwenStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset


from transformers import StoppingCriteria, StoppingCriteriaList


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.keyword_ids = []
        for keyword in keywords:
            ids = tokenizer.encode(keyword, add_special_tokens=False)
            if len(ids) > 0:
                self.keyword_ids.append(torch.tensor(ids))
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        input_ids = input_ids.cpu()
        for seq in input_ids:
            for keyword_id in self.keyword_ids:
                if len(seq) >= len(keyword_id) and torch.all((keyword_id == seq[-len(keyword_id):])).item():
                    return True
        return False
