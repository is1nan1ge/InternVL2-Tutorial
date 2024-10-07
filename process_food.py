import json
input_path = "/root/huggingface/FoodieQA/FoodieQA/sivqa_tidy.json"  # sivqa_tidy.json所在位置
output_path = "/root/huggingface/FoodieQA/FoodieQA/sivqa_llava.json"  # 输出文件位置

with open(input_path, 'r', encoding='utf-8') as f:
    foodqa = json.load(f)

llava_format = []
for data in foodqa:
    llava_format.append({
        "image": data['food_meta']['food_file'],
        "conversations": [
            {
                "from": "human",
                "value": data['question']+"\n<image>"
            },
            {
                "from": "gpt",
                "value": data['choices'][int(data['answer'])] + "，图中的菜是"+ data['food_meta']['food_name']
            }
        ]
    })

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(llava_format, f, indent=4, ensure_ascii=False)