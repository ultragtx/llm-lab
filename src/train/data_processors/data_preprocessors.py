import sys
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union

from .template import get_template_and_fix_tokenizer
import json

IGNORE_INDEX = -100


def build_history_and_prompt(context):
    if isinstance(context,str):
        context = json.loads(context)
    
    history_and_prompt = []
    pair = []
    i = 0
    for turn in context:
        if turn["role"] in ["user","return"]:
            # prompt += f"[Round {i}]\n\n"
            i += 1
            pair = []
            history_and_prompt.append(pair)
        if turn["role"] in ["user","assistant"]:
            prompt = turn["role"] + ": " + turn["content"]

            pair.append(prompt)
        else:
            if turn["role"] == "search":
                obj = turn["arguments"]
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                prompt = turn["role"] + ":\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False)
            else:
                obj = turn["records"]
                prompt = turn["role"] + ":\n" + json.dumps(obj,indent=4,ensure_ascii=False)   

            pair.append(prompt)

    history = history_and_prompt[:-1]
    prompt = history_and_prompt[-1][0]
            
    return history, prompt

def build_response(response):
    if isinstance(response,str):
        response = json.loads(response)
    
    if response["role"] == "assistant":
        return "assistant: " + response["content"]
    else:
        obj = response["arguments"]
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        return "search:\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False)


class Preprocessor:
    
    def __init__(self,data_args,tokenizer):
        self.prompt_column = data_args.prompt_column
        self.response_column = data_args.response_column
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss

        self.data_args = data_args
        self.template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    

    def construct_example(self, examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        print('-------------')
        print(len(examples[self.prompt_column]))
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                context, response = examples[self.prompt_column][i], examples[self.response_column][i]

                context = json.loads(context)
                response = json.loads(response)

                # print(context)
                # print(response)

                history, query = build_history_and_prompt(context)
                response = build_response(response)
                system = None

                yield query, response, history, system

    # 处理测试(dev/test)数据
    '''
        测试数据的拼接方式：[pad][pad]...[gmask_token][sop_token]输入文本[pad][pad]....输出文本
    '''
    def preprocess_function_eval(self,examples):  
        pass


    # 处理训练(train)数据
    '''
        训练数据的拼接方式：[gmask_token][sop_token]输入文本输出文本[eos_token][pad][pad]....
    '''
    def preprocess_function_train(self, examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in self.construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(self.template.encode_multiturn(
                self.tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(self.data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(self.data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if self.data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and self.template.efficient_eos:
                    source_mask = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]

            if len(input_ids) > self.data_args.cutoff_len:
                input_ids = input_ids[:self.data_args.cutoff_len]
                labels = labels[:self.data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs