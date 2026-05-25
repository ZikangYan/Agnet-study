# 该项目为报错管理，当发生某种情况的错误的时候，应该如何进行报错的管理，在这里实现 -> 这点对于整个程序的健壮性还是比较重要的
"""异常体系"""

class HelloAgentsException(Exception):
    """HelloAgents基础异常类"""
    pass

class LLMException(HelloAgentsException):
    """LLM相关异常"""
    pass

class AgentException(HelloAgentsException):
    """Agent相关异常"""
    pass

class ConfigException(HelloAgentsException):
    """配置相关异常"""
    pass

class ToolException(HelloAgentsException):
    """工具相关异常"""
    pass
