from abc import ABC, abstractclassmethod
from openai import OpenAI

# LLM 基类， ABC 申明此为抽象基类不能被直接实例化
class BaseLLM(ABC): 
    def __init__(
        self,
        model:str | None = None, 
        api_key: str | None = None, 
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        provider: str = "auto",
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self._client = self._init_client()
    
    @abstractclassmethod
    def _init_client(self):
        "初始化客户端"
        pass

    @abstractclassmethod
    def _chat(self):
        "定义聊天"
        pass
    

class MyLLM(BaseLLM):
    def __init__(
        self,
        model:str | None = None, 
        api_key: str | None = None, 
        base_url: str | None = None,
        provider: str = "auto",
        **kwargs
    ):
        if provider == "xxx":
            pass
        else:
            super().__init__(model, api_key, base_url, provider)
    
    def _init_client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,
        )
    
    def _chat(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model= self.model,
            messages=messages,
        )
        return resp.choices[0].message.content
    
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None