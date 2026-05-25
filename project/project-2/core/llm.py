import os
import asyncio
from typing import Optional, Iterator, List, Dict, Union, Any, AsyncIterator

from .exceptions import HelloAgentsException
from .llm_response import LLMResponse, StreamStats, LLMToolResponse
from .llm_adapters import create_adapter, BaseLLMAdapter
"""
这里是Hello-Agents的对应的调用方法，主要的工作就是对于
1. 工具调用
2. 不同的llm厂家调用
3. 调用流程匹配
核心就是 Agents 的大脑，LLM的调用
"""
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
        self._adapter: BaseLLMAdapter = create_adapter(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            model=self.model
        )
        
        # 最后调用信息统计（用于流式调用）
        self.last_call_stats : Optional[StreamStats]

    def think(self, messages: List[Dict[str,str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段

        Note:
            流式调用结束后，可通过 llm.last_call_stats 获取统计信息
        """
        print(f"🧠 正在调用 {self.model} 模型...")

        # 参数准备
        # 准备参数
        kwargs = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        try:
            print("✅ 大语言模型响应成功:")
            for chunk in self._adapter.stream_invoke(messages, **kwargs):
                print(chunk, end="", flush=True)
                yield chunk #向上传递每一个块
            print()  # 换行

            # 保存统计信息
            if hasattr(self._adapter, 'last_stats'):
                self.last_call_stats = self._adapter.last_stats

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise


    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        非流式调用LLM，返回完整响应对象。

        Args:
            messages: 消息列表
            **kwargs: 其他参数（temperature, max_tokens等）

        Returns:
            LLMResponse: 包含内容、统计信息、推理过程（thinking model）的响应对象

        Example:
            response = llm.invoke([{"role": "user", "content": "你好"}])
            print(response.content)  # 回复内容
            print(response.usage)    # token使用量
            print(response.latency_ms)  # 耗时
            if response.reasoning_content:  # thinking model的推理过程
                print(response.reasoning_content)
        """
        # 合并参数
        call_kwargs = {
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = kwargs.pop("max_tokens", self.max_tokens)
        call_kwargs.update(kwargs)

        return self._adapter.invoke(messages, **call_kwargs)

    def stream_invoke(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            str: 流式响应的文本片段

        Note:
            流式调用结束后，可通过 llm.last_call_stats 获取统计信息
        """
        temperature = kwargs.pop("temperature", None)

        # 准备参数
        call_kwargs = {}
        if self.max_tokens:
            call_kwargs["max_tokens"] = kwargs.pop("max_tokens", self.max_tokens)
        call_kwargs.update(kwargs)

        for chunk in self._adapter.stream_invoke(messages, temperature=temperature, **call_kwargs):
            yield chunk

        # 保存统计信息
        if hasattr(self._adapter, 'last_stats'):
            self.last_call_stats = self._adapter.last_stats

    def invoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs
    ) -> LLMToolResponse:
        """
        调用 LLM 并支持工具调用（Function Calling）

        这是支持 OpenAI Function Calling 的核心方法，用于结构化工具调用。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            tools: 工具 schema 列表，格式为 OpenAI Function Calling 规范
            tool_choice: 工具选择策略
                - "auto": 让模型自动决定是否调用工具（默认）
                - "none": 强制不调用工具
                - "required": 强制调用工具
                - {"type": "function", "function": {"name": "tool_name"}}: 强制调用指定工具
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            统一的工具调用响应对象 (LLMToolResponse)

        Raises:
            HelloAgentsException: 当 LLM 调用失败时
        """
        # 合并参数
        call_kwargs = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "tool_choice": tool_choice,
        }
        if self.max_tokens:
            call_kwargs["max_tokens"] = kwargs.pop("max_tokens", self.max_tokens)
        call_kwargs.update(kwargs)

        return self._adapter.invoke_with_tools(messages, tools, **call_kwargs)

    # ==================== 异步方法 ====================

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        异步非流式调用 LLM

        在线程池中运行同步 invoke 方法，避免阻塞事件循环

        Args:
            messages: 消息列表
            **kwargs: 其他参数（temperature, max_tokens等）

        Returns:
            LLMResponse: 包含内容、统计信息的响应对象

        Example:
            response = await llm.ainvoke([{"role": "user", "content": "你好"}])
            print(response.content)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke(messages, **kwargs)
        )

    async def astream_invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        真正的异步流式调用 LLM（使用 adapter 的异步实现）

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            str: 流式响应的文本片段（实时返回）

        Example:
            async for chunk in llm.astream_invoke(messages):
                print(chunk, end="", flush=True)
        """
        # 使用 adapter 的异步流式方法
        async for chunk in self._adapter.astream_invoke(messages, **kwargs):
            yield chunk

        # 保存统计信息
        if hasattr(self._adapter, 'last_stats'):
            self.last_call_stats = self._adapter.last_stats

    async def ainvoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs
    ) -> LLMToolResponse:
        """
        异步调用 LLM 并支持工具调用（Function Calling）

        Args:
            messages: 消息列表
            tools: 工具 schema 列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Returns:
            统一的工具调用响应对象 (LLMToolResponse)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke_with_tools(messages, tools, tool_choice, **kwargs)
        )
# 上面还是调用了各种的方法