from datasets import Dataset
from langchain_core.documents import Document
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate
from typing import List, Dict


class RagEvaluator:
    def __init__(self):
        # 데이터 저장을 위한 리스트 초기화
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.contexts: List[List[Document]] = []

    def add_sample(self, question: str, answer: str, context: List[Document]):
        """평가할 데이터 샘플을 추가합니다."""
        self.questions.append(question)
        self.answers.append(answer)
        context_list = [doc.page_content for doc in context]
        self.contexts.append(context_list)

    def get_samples(self) -> Dict:
        """현재까지 저장된 모든 샘플을 딕셔너리 형태로 반환합니다."""
        return {
            "question": self.questions,
            "answer": self.answers,
            "contexts": self.contexts,
        }

    def evaluate_all(self):
        """저장된 데이터에 대해 RAG 평가를 수행합니다."""
        if not self.questions:
            raise ValueError(
                "평가할 데이터가 없습니다. add_sample()을 통해 데이터를 먼저 추가해주세요."
            )

        # Dataset 생성
        dataset = Dataset.from_dict(self.get_samples())

        # 평가 수행
        score = evaluate(dataset, metrics=[answer_relevancy, faithfulness])

        return score.to_pandas()

    def evaluate_last(self):
        """마지막 샘플에 대해 RAG 평가를 수행합니다."""
        if not self.questions:
            raise ValueError(
                "평가할 데이터가 없습니다. add_sample()을 통해 데이터를 먼저 추가해주세요."
            )

        last_sample = {
            "question": [self.get_samples()["question"][-1]],
            "answer": [self.get_samples()["answer"][-1]],
            "contexts": [self.get_samples()["contexts"][-1]],
        }

        dataset = Dataset.from_dict(last_sample)
        score = evaluate(dataset, metrics=[answer_relevancy, faithfulness])
        return score.to_pandas()

    def clear(self):
        """평가 데이터를 초기화합니다."""
        self.questions = []
        self.answers = []
        self.contexts = []
