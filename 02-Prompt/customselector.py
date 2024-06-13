from langchain_core.example_selectors.base import BaseExampleSelector
import numpy as np


class CustomExampleSelector(BaseExampleSelector):
    """
    입력된 텍스트에 가장 유사한 예제들을 선택하는 클래스입니다.
    이 클래스는 OpenAI의 임베딩 모델을 사용하여 예제들의 벡터 표현을 사전 계산하고,
    입력된 텍스트와 예제들 사이의 코사인 유사도를 기반으로 가장 유사한 예제들을 선택합니다.

    Attributes:
        examples (list): 선택 기준이 될 예제 목록.
        embedding_model (object): 텍스트를 벡터로 변환하는 임베딩 모델.
        search_key (str): 예제에서 입력 텍스트와 비교할 키 값.
    """

    def __init__(self, examples, embedding_model, search_key="instruction"):
        """
        예제 목록, 임베딩 모델, 검색 키를 초기화합니다.

        Args:
            examples (list): 예제 데이터 목록.
            embedding_model (object): 임베딩을 계산할 모델.
            search_key (str): 예제에서 입력과 비교할 때 사용할 키.
        """
        self.examples = examples
        self.embedding_model = embedding_model
        self.search_key = search_key
        # 모든 예제에 대한 임베딩을 사전 계산합니다.
        self.example_embeddings = [
            (example, self.embedding_model.embed_query(example[search_key]))
            for example in examples
        ]

    def cosine_similarity(self, vec1, vec2):
        """두 벡터 간의 코사인 유사도를 계산합니다."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def add_example(self, example):
        """예제 목록에 새 예제를 추가합니다."""
        self.examples.append(example)

    def select_examples(self, input_variables, k=1):
        """
        주어진 입력 변수에 대해 가장 유사한 k개의 예제를 선택합니다.

        Args:
            input_variables (dict): 검색 키와 함께 입력 텍스트를 포함하는 사전.
            k (int): 반환할 최상위 예제의 수.

        Returns:
            list: 가장 유사도가 높은 k개의 예제.
        """
        # 입력 텍스트의 임베딩을 계산합니다.
        input_text = input_variables[self.search_key]
        input_embedding = self.embedding_model.embed_query(input_text)

        # 유사도를 계산하고 예제들과 함께 저장합니다.
        similarities = []
        for example, example_embedding in self.example_embeddings:
            similarity = self.cosine_similarity(input_embedding, example_embedding)
            similarities.append((example, similarity))

        # 유사도가 높은 순서대로 예제들을 정렬합니다.
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 유사도가 가장 높은 상위 k개의 예제를 반환합니다.
        return [example for example, _ in similarities[:k]]
