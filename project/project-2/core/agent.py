from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING, AsyncGenerator
import asyncio
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from .lifecycle import AgentEvent, EventType, LifecycleHook, ExecutionContext

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry
    from ..observability.trace_logger import TraceLogger
    from ..tools.tool_filter import ToolFilter

class Agent(ABC):
    """Agent基类

    集成能力：
    - HistoryManager: 历史管理与压缩
    - ObservationTruncator: 工具输出截断
    - TraceLogger: 可观测性（JSONL + HTML）
    - ToolRegistry: 工具管理（可选）
    - SkillLoader: 知识外化（可选）

    向后兼容：
    - self._history 属性仍然可用（通过 property 代理）
    - add_message/clear_history/get_history 方法保持不变
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()

        # 工具注册表（可选）
        self.tool_registry = tool_registry # 工具注册表是如何实现的？