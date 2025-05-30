import os
import torch
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import AutoPeftModelForCausalLM


def init_model():
    print("init model ...")
    path = "/home/xym/othercode/base_model/baichuan2_chat"
    baichuan_resume_path = "/home/xym/mycode/law_LLM/cot_law/baichuan_outout/cot/checkpoint-218"
    model = AutoPeftModelForCausalLM.from_pretrained(baichuan_resume_path, trust_remote_code=True,device_map='cuda:3')
    tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
            trust_remote_code=True
        )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用擎盾数据，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model, tokenizer = init_model()
    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n案情信息：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\n擎盾数据：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        prompt = "假设你是一名法院法官，请根据输入的裁判文书和刑法法条,预测被告人的罪名,被告人只有一种罪名。请一步一步分析。" + prompt
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
