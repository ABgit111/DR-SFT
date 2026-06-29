from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model.config._name_or_path, add_eos_token=False, add_bos_token=False
    )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = padding_side

    return model, tokenizer


class QwenStringConverter:
    def string_formatter(example):
        """Parse OpenAI style chat format to Qwen2 format for fine-tuning."""

        IM_START, IM_END = "<|im_start|>", "<|im_end|>"
        ENDOFTEXT = "<|endoftext|>"

        if "messages" not in example:
            raise ValueError("No messages in the example")

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")

        pt = 0
        str_message = ""

        # Handle system prompt
        if messages[0]["role"] == "system":
            str_message = f"{IM_START}system\n{messages[0]['content']}{IM_END}\n"
            pt = 1
        else:
            str_message = f"{IM_START}system\n{default_system_prompt}{IM_END}\n"

        if pt == len(messages):
            raise ValueError("The message should be user - assistant alternation")

        while pt < len(messages):
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")

            str_message += f"{IM_START}user\n{messages[pt]['content']}{IM_END}\n"
            pt += 1

            if pt >= len(messages):
                raise ValueError("The message should be user - assistant alternation")

            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")

            str_message += f"{IM_START}assistant\n{messages[pt]['content']}{IM_END}"
            pt += 1

            if pt == len(messages):
                str_message += ENDOFTEXT
            else:
                str_message += "\n"

        return {"text": str_message}

    def string_formatter_completion_only(example):
        """Parse OpenAI style chat format to Qwen2 format for inference."""

        IM_START, IM_END = "<|im_start|>", "<|im_end|>"

        if "messages" not in example:
            raise ValueError("No messages in the example")

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")

        pt = 0
        str_message = ""

        # Handle system prompt
        if messages[0]["role"] == "system":
            str_message = f"{IM_START}system\n{messages[0]['content']}{IM_END}\n"
            pt = 1
        else:
            str_message = f"{IM_START}system\n{default_system_prompt}{IM_END}\n"

        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")

            str_message += f"{IM_START}user\n{messages[pt]['content']}{IM_END}\n"
            pt += 1

            if pt >= len(messages) - 1:
                break

            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")

            str_message += f"{IM_START}assistant\n{messages[pt]['content']}{IM_END}\n"
            pt += 1

        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "Completion only mode should end with a header of assistant message"
            )

        str_message += f"{IM_START}assistant\n{messages[-1]['content']}"

        return {"text": str_message}

    def conversion_to_qwen_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(
            QwenStringConverter.string_formatter, remove_columns=redundant_columns
        )
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
                if (
                    len(seq) >= len(keyword_id)
                    and torch.all((keyword_id == seq[-len(keyword_id) :])).item()
                ):
                    return True
        return False


def get_qwen_stopping_criteria(tokenizer):
    stop_keywords = ["<|im_end|>", "<|endoftext|>"]
    return StoppingCriteriaList(
        [KeywordStoppingCriteria(keywords=stop_keywords, tokenizer=tokenizer)]
    )
