from typing import Any, List, TypeVar, Generic
from langchain_teddynote.tools.tavily import TavilySearch
from abc import ABC, abstractmethod

T = TypeVar("T")  # 제네릭 타입 변수 정의


class BaseTool(ABC, Generic[T]):
    """도구들의 기본 클래스"""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """도구 초기화를 위한 추상 메서드"""
        pass

    @abstractmethod
    def _create_tool(self) -> T:
        """실제 도구 객체를 생성하는 추상 메서드"""
        pass

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> T:
        """도구 객체를 생성하고 바로 반환하는 팩토리 메서드"""
        instance = cls(*args, **kwargs)
        tool = instance._create_tool()
        return tool

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """도구를 실행하는 추상 메서드"""
        pass


class WebSearchTool(BaseTool[TavilySearch]):
    """웹 검색을 수행하는 도구 클래스"""

    def __init__(
        self,
        topic: str = "general",
        max_results: int = 3,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        format_output: bool = False,
        include_domains: List[str] = [],
        exclude_domains: List[str] = [],
    ):
        """WebSearchTool 초기화 메서드"""
        super().__init__()
        self.topic = topic
        self.max_results = max_results
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.format_output = format_output
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains

    def _create_tool(self) -> TavilySearch:
        """TavilySearch 객체를 생성하고 설정하는 내부 메서드"""
        search = TavilySearch(
            topic=self.topic,
            max_results=self.max_results,
            include_answer=self.include_answer,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            format_output=self.format_output,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
        )
        search.name = "web_search"
        search.description = "Use this tool to search on the web"
        return search

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """도구를 실행하는 메서드"""
        tool = self._create_tool()
        return tool(*args, **kwargs)
