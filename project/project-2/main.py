from core.llm import MyLLM

api_key="sk-7ff1bb81362f4d4a83bd00f0a1d5b822"
base_url="https://api.deepseek.com"
model="deepseek-chat"
llm = MyLLM(model, api_key, base_url, provider="deepseek")
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
print(llm._chat(messages))
print(llm.think(messages))
