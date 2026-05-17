"""AgentKB 自定义异常层次结构。"""


class AgentKBException(Exception):
    """所有自定义异常的基类。"""


class ConfigError(AgentKBException):
    """配置加载或校验失败。"""


class LLMConnectionError(AgentKBException):
    """无法连接到大模型后端（Ollama/OpenAI）。"""


class LLMResponseError(AgentKBException):
    """大模型返回了无效或为空的响应。"""


class EmbeddingError(AgentKBException):
    """向量化模型编码文本失败。"""


class KnowledgeBaseError(AgentKBException):
    """知识库操作（上传/检索/删除）失败。"""


class ToolExecutionError(AgentKBException):
    """工具执行失败——非致命，Agent 可降级处理。"""


class SessionError(AgentKBException):
    """会话持久化或读取失败。"""
