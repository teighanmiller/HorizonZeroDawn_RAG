import json
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat, ChatResponse
from sentence_transformers import SentenceTransformer
from src.rag_chat import Chat

COLLECTION_NAME = "horizon_rag"


def precision(items: list, relevant_items: list, k: int = 1) -> float:
    relevant_cnt = 0
    for cnt in range(k):
        if items[cnt] in relevant_items:
            relevant_cnt += 1
    return float(relevant_cnt / k)


def recall(items: list, relevant_items: list, k: int = 1) -> float:
    recall_cnt = 0
    for i in range(k):
        if relevant_items[i] in items:
            recall_cnt += 1
    return float(recall_cnt / k)


def f1_score(items: list, relevant_items: list, k: int = 1) -> float:
    k_precision = precision(items, relevant_items, k)
    k_recall = recall(items, relevant_items, k)
    if k_precision == 0 and k_recall == 0:
        return 0
    return (2 * k_precision * k_recall) / (k_precision + k_recall)


def first_relevant_item(items: list, relevant_items: list, k: int = 1) -> float:
    for idx, item in enumerate(items):
        if item in relevant_items:
            return 1 / (idx + 1)

        if idx == k:
            return 0


def get_cosine_similarity(
    query: str, ground_truth: str, transformer: SentenceTransformer
) -> float:
    query_enc = transformer.encode(query)
    truth_enc = transformer.encode(ground_truth)
    return cosine_similarity(query_enc.reshape(1, -1), truth_enc.reshape(1, -1)).item()


def answer_relevancy(
    original_question: str, generated_answer: str, transformer: SentenceTransformer
) -> float:
    score = 0
    response: ChatResponse = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": f"Based on this answer {generated_answer}, generate 3 possible questions this could have been in response to. Return only the questions, seperate each question by two line breaks.",
            },
        ],
    )

    questions = response["message"]["content"].split("\n\n")
    q_embed = transformer.encode(original_question)
    for query in questions:
        score += cosine_similarity(
            q_embed.reshape(1, -1),
            transformer.encode(query.strip()).reshape(1, -1),
        ).item()
    return score / len(questions)


def evaluate_retrieval(retrieved_context: list, relevant_context: list) -> str:
    evaluation = {}
    evaluation["Precision"] = precision(retrieved_context, relevant_context)
    evaluation["Recall"] = recall(retrieved_context, relevant_context)
    evaluation["F1_Score"] = f1_score(retrieved_context, relevant_context)
    return evaluation


def evaluate_generation(
    question: str, ground_truth: str, answer: str, transformer: SentenceTransformer
) -> str:
    evaluation = {}
    evaluation["Cosine Similarity"] = get_cosine_similarity(
        answer, ground_truth, transformer
    )
    evaluation["Answer Relevancy"] = answer_relevancy(question, answer, transformer)
    return evaluation


def evaluate_rag(filepath: str) -> str:
    new_chat = Chat(COLLECTION_NAME)

    with open(filepath, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    evaluated_content = []
    mrr = 0

    for item in test_data:
        question = item["Question"]
        sys_reword_prompt, user_reword_prompt = new_chat.get_reword_prompt(
            question, use_history=False
        )

        reword_dict = json.loads(
            new_chat.get_llm_response(
                system_prompt=sys_reword_prompt, user_query=user_reword_prompt
            )
        )

        retrieved_content = new_chat.retrieval(
            query=question, classification=reword_dict["classification"]
        )[:3]

        mrr += first_relevant_item(
            [point.id for point in retrieved_content],
            [list(context.keys())[0] for context in item["relevant"]],
        )

        sys_query_prompt, user_query_prompt = new_chat.get_rag_prompt(
            reword_dict["query"],
            [pnt.payload["content"] for pnt in retrieved_content],
        )

        response = new_chat.get_llm_response(
            system_prompt=sys_query_prompt, user_query=user_query_prompt
        )

        retrieval_evaluation = evaluate_retrieval(
            [list(context.keys())[0] for context in item["relevant"]],
            [point.id for point in retrieved_content],
        )

        generation_evaluation = evaluate_generation(
            question=question,
            ground_truth=item["Answer"],
            answer=response,
            transformer=new_chat.dense_model,
        )

        evaluation = item | retrieval_evaluation | generation_evaluation
        evaluated_content.append(evaluation)

    mrr /= len(test_data)
    evaluated_content.append({"Mean Recipricol Rank": mrr})

    with open(
        "/Users/teighanmiller/development/courses/zoomcamp/HorizonZeroDawn_RAG/data/evaluated_data.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(evaluated_content, f, indent=4)


if __name__ == "__main__":
    file = "/Users/teighanmiller/development/courses/zoomcamp/HorizonZeroDawn_RAG/data/evaluation_data.json"
    evaluate_rag(file)
