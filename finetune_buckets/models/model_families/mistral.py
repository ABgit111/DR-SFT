from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

default_system_prompt = ""


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True

    tokenizer = AutoTokenizer.from_pretrained(
        model.config._name_or_path, add_eos_token=False, add_bos_token=False
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = padding_side

    return model, tokenizer


class MistralStringConverter:
    def string_formatter(example):
        """Parse OpenAI style chat format to Mistral format for fine-tuning."""

        BOS, EOS = "<s>", "</s>"
        B_INST, E_INST = "[INST] ", " [/INST] "

        if "messages" not in example:
            raise ValueError("No messages in the example")

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")

        pt = 0
        # Handle system prompt
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            pt = 1
        else:
            system_prompt = default_system_prompt

        if pt == len(messages):
            raise ValueError("The message should be user - assistant alternation")

        str_message = ""

        while pt < len(messages):
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")

            # For Mistral, system prompt is incorporated into the first user message
            if pt == 0 or (pt == 1 and messages[0]["role"] == "system"):
                if system_prompt:
                    str_message += (
                        BOS
                        + B_INST
                        + system_prompt
                        + "\n\n"
                        + messages[pt]["content"]
                        + E_INST
                    )
                else:
                    str_message += BOS + B_INST + messages[pt]["content"] + E_INST
            else:
                str_message += BOS + B_INST + messages[pt]["content"] + E_INST

            pt += 1

            if pt >= len(messages):
                raise ValueError("The message should be user - assistant alternation")

            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")

            str_message += messages[pt]["content"] + EOS
            pt += 1

        return {"text": str_message}

    def string_formatter_completion_only(example):
        """Parse OpenAI style chat format to Mistral format for inference."""

        BOS, EOS = "<s>", "</s>"
        B_INST, E_INST = "[INST] ", " [/INST] "

        if "messages" not in example:
            raise ValueError("No messages in the example")

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")

        pt = 0
        # Handle system prompt
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            pt = 1
        else:
            system_prompt = default_system_prompt

        str_message = ""

        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("The message should be user - assistant alternation")

            # For Mistral, system prompt is incorporated into the first user message
            if pt == 0 or (pt == 1 and messages[0]["role"] == "system"):
                if system_prompt:
                    str_message += (
                        BOS
                        + B_INST
                        + system_prompt
                        + "\n\n"
                        + messages[pt]["content"]
                        + E_INST
                    )
                else:
                    str_message += BOS + B_INST + messages[pt]["content"] + E_INST
            else:
                str_message += BOS + B_INST + messages[pt]["content"] + E_INST

            pt += 1

            if pt >= len(messages) - 1:
                break

            if messages[pt]["role"] != "assistant":
                raise ValueError("The message should be user - assistant alternation")

            str_message += messages[pt]["content"] + EOS
            pt += 1

        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "Completion only mode should end with a header of assistant message"
            )

        str_message += messages[-1]["content"]

        return {"text": str_message}

    def conversion_to_mistral_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(
            MistralStringConverter.string_formatter, remove_columns=redundant_columns
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


def get_mistral_stopping_criteria(tokenizer):
    stop_keywords = ["</s>", "[/INST]"]
    return StoppingCriteriaList(
        [KeywordStoppingCriteria(keywords=stop_keywords, tokenizer=tokenizer)]
    )
