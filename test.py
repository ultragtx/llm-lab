

import json

def build_prompt(context):
    
    if isinstance(context,str):
        context = json.loads(context)
    
    prompt = ""
    i = 0
    for turn in context:
        if turn["role"] in ["user","return"]:
            prompt += f"[Round {i}]\n\n"
            i += 1
        if turn["role"] in ["user","assistant"]:
            prompt += turn["role"] + ": " + turn["content"] + "\n\n"
        else:
            if turn["role"] == "search":
                obj = turn["arguments"]
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                prompt += turn["role"] + ":\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + "\n\n"
            else:
                obj = turn["records"]
                prompt += turn["role"] + ":\n" + json.dumps(obj,indent=4,ensure_ascii=False) + "\n\n"   
            
    return prompt

def build_prompt2(context):
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



a = {"context": "[{\"role\": \"user\", \"content\": \"你好，帮我推荐一个能提供早餐服务的酒店，谢谢。\"}, {\"role\": \"search\", \"arguments\": {\"facilities\": [\"早餐服务\"]}}, {\"role\": \"return\", \"records\": [{\"name\": \"北京新世界酒店\", \"type\": \"豪华型\", \"address\": \"北京东城区祈年大街8号\", \"subway\": \"崇文门地铁站D口\", \"phone\": \"010-59608888\", \"facilities\": [\"公共区域和部分房间提供wifi\", \"宽带上网\", \"国际长途电话\", \"吹风机\", \"24小时热水\", \"暖气\", \"西式餐厅\", \"中式餐厅\", \"残疾人设施\", \"室内游泳池\", \"会议室\", \"健身房\", \"SPA\", \"无烟房\", \"商务中心\", \"酒吧\", \"桑拿\", \"早餐服务\", \"接机服务\", \"接待外宾\", \"洗衣服务\", \"行李寄存\", \"看护小孩服务\", \"租车\", \"叫醒服务\"], \"price\": 989.0, \"rating\": 4.6, \"hotel_id\": 94}]}, {\"role\": \"assistant\", \"content\": \"给您推荐北京新世界酒店。\"}, {\"role\": \"user\", \"content\": \"没关系，那就帮我推荐一个经济型酒店吧。\"}, {\"role\": \"search\", \"arguments\": {\"facilities\": [\"早餐服务\"], \"type\": \"经济型\"}}, {\"role\": \"return\", \"records\": [{\"name\": \"汉庭酒店(北京新国展店)\", \"type\": \"经济型\", \"address\": \"北京顺义区天竺空港经济开发区A区天柱西路24号\", \"subway\": \"国展地铁站B口\", \"phone\": \"010-80481588\", \"facilities\": [\"酒店各处提供wifi\", \"宽带上网\", \"吹风机\", \"24小时热水\", \"中式餐厅\", \"会议室\", \"无烟房\", \"商务中心\", \"早餐服务\", \"接机服务\", \"行李寄存\", \"叫醒服务\"], \"price\": 209.0, \"rating\": 4.4, \"hotel_id\": 536}]}, {\"role\": \"assistant\", \"content\": \"为您推荐汉庭酒店(北京新国展店)。\"}, {\"role\": \"user\", \"content\": \"这家酒店的价格是多少？\"}]", "response": "{\"role\": \"assistant\", \"content\": \"北京新世界酒店的每晚房价是989元。\"}"}
a = {"context": "[{\"role\": \"user\", \"content\": \"你好，请帮我推荐一个提供无烟房的舒适型酒店可以吗？\"}, {\"role\": \"search\", \"arguments\": {\"facilities\": [\"无烟房\"], \"type\": \"舒适型\"}}, {\"role\": \"return\", \"records\": [{\"name\": \"北京红驿栈酒店\", \"type\": \"舒适型\", \"address\": \"北京朝阳区东直门外春秀路太平庄10号(主副楼在一幢建筑里)\", \"subway\": \"东直门地铁站E口\", \"phone\": \"010-64171066\", \"facilities\": [\"公共区域和部分房间提供wifi\", \"宽带上网\", \"国际长途电话\", \"吹风机\", \"24小时热水\", \"暖气\", \"无烟房\", \"早餐服务\", \"接待外宾\", \"行李寄存\", \"叫醒服务\"], \"price\": 344.0, \"rating\": 4.7, \"hotel_id\": 51}, {\"name\": \"维也纳国际酒店(北京广安门店)\", \"type\": \"舒适型\", \"address\": \"北京西城区白广路7号\", \"subway\": \"广安门内地铁站C口\", \"phone\": \"010-83539988\", \"facilities\": [\"酒店各处提供wifi\", \"宽带上网\", \"吹风机\", \"24小时热水\", \"中式餐厅\", \"会议室\", \"无烟房\", \"商务中心\", \"早餐服务\", \"洗衣服务\", \"行李寄存\", \"叫醒服务\"], \"price\": 553.0, \"rating\": 4.7, \"hotel_id\": 56}]}]", "response": "{\"role\": \"assistant\", \"content\": \"没问题，推荐你去北京红驿栈酒店和维也纳国际酒店(北京广安门店)，都挺好的。\"}"}
# a = {"context": "[{\"role\": \"user\", \"content\": \"你好，请帮我推荐一个提供无烟房的舒适型酒店可以吗？\"}]", "response": "{\"role\": \"search\", \"arguments\": {\"facilities\": [\"无烟房\"], \"type\": \"舒适型\"}}"}


# print(a["context"])
history, prompt = build_prompt2(a["context"])
for p in history:
    print(p)

print(prompt)

print('----------------')
print(build_response(a["response"]))