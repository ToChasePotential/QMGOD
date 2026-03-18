
def step2(*, x_que: str, **kwargs):
    """
    Pipeline-visible Step2 wrapper
    """
    return build_question_graph(x_que)

print("[LOADED] step2_question_graph from:", __file__)

import re
from typing import Dict, Any, List


def _extract_capitalized_words(text: str) -> List[str]:
    return re.findall(r"[A-Z][a-zA-Z]{2,}", text or "")


def _extract_keywords(text: str) -> List[str]:

    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stopwords = {
        "what", "which", "where", "when", "that",
        "about", "with", "from", "this", "these",
        "those", "whose", "does", "is", "are",
    }
    return [w for w in words if w not in stopwords]


def build_question_graph(x_que: str) -> Dict[str, Any]:

    low_entities = []
    low_relations = []

    low_words = _extract_capitalized_words(x_que)
    for i, w in enumerate(low_words):
        low_entities.append(
            {
                "id": f"qL_e{i}",
                "name": w,
                "type": "QuestionEntity",
            }
        )
        if i > 0:
            low_relations.append(
                {
                    "id": f"qL_r{i-1}",
                    "head": f"qL_e{i-1}",
                    "tail": f"qL_e{i}",
                    "type": "question_co_occurrence",
                }
            )


    high_entities = []
    high_relations = []

    high_words = _extract_keywords(x_que)
    for i, w in enumerate(high_words[:5]):  # cap size on purpose
        high_entities.append(
            {
                "id": f"qH_e{i}",
                "name": w,
                "type": "QuestionTopic",
            }
        )

        if i > 0:
            high_relations.append(
                {
                    "id": f"qH_r{i-1}",
                    "head": f"qH_e{i-1}",
                    "tail": f"qH_e{i}",
                    "type": "semantic_flow",
                }
            )

    return {
        "low": {
            "E": low_entities,
            "R": low_relations,
        },
        "high": {
            "E": high_entities,
            "R": high_relations,
        },
    }


if __name__ == "__main__":
    q = "What country is Paris located in and which continent does it belong to?"
    g = build_question_graph(q)
    print(g)
