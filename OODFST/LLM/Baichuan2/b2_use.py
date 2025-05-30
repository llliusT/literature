from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import tqdm


path = "/home/xym/othercode/base_model/baichuan2_chat"
checkpoint_paths = ["/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_output/top5_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_output/top8_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_output/top10_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_output/top5_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_output/top8_ave/checkpoint-625",
                   "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_output/top8_ave/checkpoint-625",
                   "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_output/top_5_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_output/top_8_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_output/top_10_ave/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/wos/baichuan_output/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/wos/output/top8/checkpoint-1250",
                   "/home/xym/mycode/Out-of-Domain/wos/output/top10/checkpoint-1250",

                   ]

data_paths = ["/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_data/new_test_top5.json",
             "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_data/new_test_top8.json",
             "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_data/new_test_top10.json",
             "/home/xym/mycode/Out-of-Domain/medical_data/test_data/new_test_top5.json",
             "/home/xym/mycode/Out-of-Domain/medical_data/test_data/new_test_top8.json",
             "/home/xym/mycode/Out-of-Domain/medical_data/test_data/new_test_top10.json",
             "/home/xym/mycode/Out-of-Domain/railway_data/test_data/new_test_top5.json",
             "/home/xym/mycode/Out-of-Domain/railway_data/test_data/new_test_top8.json",
             "/home/xym/mycode/Out-of-Domain/railway_data/test_data/new_test_top10.json",
             "/home/xym/mycode/Out-of-Domain/wos/test_data/new_test_top5.json",
             "/home/xym/mycode/Out-of-Domain/wos/test_data/new_test_top8.json",
             "/home/xym/mycode/Out-of-Domain/wos/test_data/new_test_top10.json",
             ]

save_paths = ["/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_pre/new_test_top5.json",
              "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_pre/new_test_top8.json",
              "/home/xym/mycode/Out-of-Domain/agriculture_data/baichuan_pre/new_test_top10.json",
              "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_pre/new_test_top5.json",
              "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_pre/new_test_top8.json",
              "/home/xym/mycode/Out-of-Domain/medical_data/baichuan_pre/new_test_top10.json",
              "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_pre/new_test_top5.json",
              "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_pre/new_test_top8.json",
              "/home/xym/mycode/Out-of-Domain/railway_data/baichuan_pre/new_test_top10.json",
              "/home/xym/mycode/Out-of-Domain/wos/baichuan_pre/new_test_top5.json",
              "/home/xym/mycode/Out-of-Domain/wos/baichuan_pre/new_test_top8.json",
              "/home/xym/mycode/Out-of-Domain/wos/baichuan_pre/new_test_top10.json",
              ]
device = "cuda:2"

for i in range(len(data_paths)):
    checkpoint_path = checkpoint_paths[i]
    data_path = data_paths[i]
    save_path = save_paths[i]
    # 加载模型
    print(checkpoint_path)
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
            trust_remote_code=True
        )
    # 加载数据
    with open(data_path,'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
        #datas = json.load(f)
    # 生成预测
    output = []
    i = 1
    for data in tqdm.tqdm(datas):
        messages = []
        input = data["input"]
        res = {}
        try:
            label = data["label"]  
        except:
            label = data["output"]
        res["input"] = input
        res["label"] = label
        messages.append({"role": "user", "content": input})
        pre = ''
        position = 0
        for response in model.chat(tokenizer, messages, stream=True):
            pre += response[position:]
            position = len(response)
        res["pre"] = pre.replace("\n","")
        output.append(res)
        # print(f"finish:{i}/{len(datas)}")
        i += 1
    # 保存结果
    del model
    with open(save_path,'a',encoding='utf-8') as s:
        json.dump(output,s,indent=2,ensure_ascii=False)
    print("finish")
