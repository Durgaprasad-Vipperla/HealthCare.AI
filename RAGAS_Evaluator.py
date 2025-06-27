import pandas as pd
import os
from CanvasLLMWrapper import CanvasLLM
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness, FactualCorrectness, LLMContextPrecisionWithoutReference, AnswerRelevancy

excel_file = "chat_history.xlsx"
if not os.path.exists(excel_file):
    raise FileNotFoundError(f"File '{excel_file}' not found!")

df = pd.read_excel(excel_file, engine="openpyxl")

dataset = []

for _,row in df.iterrows():
    dataset.append({
        "user_input": row["User Question"],
        "retrieved_contexts": row["Context"].split("|||") if isinstance(row["Context"], str) else [],
        "response": row["LLM Response"]
        # "reference": row.get("Ground Truth", None)
    })

evaluation_dataset = EvaluationDataset.from_list(dataset)

llm = CanvasLLM()
evaluator_llm = LangchainLLMWrapper(llm)

metrics = [
    # LLMContextRecall(),
    # LLMContextPrecisionWithoutReference(),
    Faithfulness(),
    # AnswerRelevancy()
]

result = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=evaluator_llm)

print("RAGAS Evaluation Results:")
print(result)
