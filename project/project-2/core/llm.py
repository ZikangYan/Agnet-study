from abc import ABC, abstractclassmethod
from openai import OpenAI
from .exception import HelloAgentsException
from .llm_adapters import create_adapter, BaseLLMAdapter

class HelloAgents: 
    def __init__(
        self,
        model:str | None = None, 
        api_key: str | None = None, 
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        # provider: str = "auto",
        **kwargs
    ):
        """
        对于项目来说，要写一段项目的解释说明，我这里定制一个标准，类似于 skills 的写法，需要markdown的格式去写

        name: BaseLLM 
        description: 用于床架一个基础的LLM客户端；用于调用 大模型API
        Args: 部分参数说明
            model: 模型名称，默认从 LLM_MODEL_ID 读取
            api_key: API密钥，默认从 LLM_API_KEY 读取
            base_url: 服务地址，默认从 LLM_BASE_URL 读取
            temperature: 温度参数，默认0.7
            max_tokens: 最大token数
            timeout: 超时时间（秒），默认从 LLM_TIMEOUT 读取，默认60秒
        """
        # 加载配置-> 用于某个功能的参数放在一起，其他功能另起一行
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # 验证必要参数-> 对于某些必要项进行验证-> 重要校验还是用if ... raise 比较好
        if not self.model:
            raise HelloAgentsException("必须提供模型名称（model参数或LLM_MODEL_ID环境变量）")
        if not self.api_key:
            raise HelloAgentsException("必须提供API密钥（api_key参数或LLM_API_KEY环境变量）")
        if not self.base_url:
            raise HelloAgentsException("必须提供服务地址（base_url参数或LLM_BASE_URL环境变量）")

        # 创建适配器（这个适配器是什么）
        self._adapter: BaseLLMAdapter = 


    