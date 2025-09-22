import dspy


lm = dspy.LM("ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)


class RouterModule(dspy.Signature):
    """
    Classify questions into the appropriate processing method:
    - rag: For questions about policies, documents, guidelines, explanations, or content that would be in text files
    - sql: For questions about structured data, counts, averages, or data that would be in database tables
    - hybrid: For questions requiring both document knowledge and structured data
    """

    question: str = dspy.InputField(desc="The user's question to classify")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the classification")
    router = dspy.OutputField(desc="The classification of the question",
        choices=["rag", "sql", "hybrid"])
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")

    @staticmethod
    def _router(question: str):
        classify = dspy.Predict(RouterModule)
        return classify(question=question)
    
def validate_category(example, prediction, trace=None):
    return prediction.router == example.router

# Create a proper DSPy Module for the router
class RouterProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(RouterModule)
    
    def forward(self, question):
        return self.classify(question=question)

# Create an instance of the module
router_program = RouterProgram()

# response = router(question="During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold.")
# print(response)


import csv
import json
import dspy
from dspy.evaluate import Evaluate


# Load the trainset
trainset = []
with open('/Users/sreerajnair/my_space/my_space/QnA_Assistant_Using_DSPy_and_LangGraph-main/dataset.json', 'r') as file:
    reader = json.load(file)
    for row in reader:
        example = dspy.Example(question=row['question'], router=row['route']).with_inputs("question") 
        trainset.append(example)

print(f"Loaded {len(trainset)} training examples.")
print(trainset)

# Evaluate our existing function
evaluator = Evaluate(devset=trainset, num_threads=1, display_progress=True, display_table=5)
evaluator(router_program, metric=validate_category)


# Now use this module with MIPROv2
tp = dspy.MIPROv2(metric=validate_category, auto="light")
optimized_classify = tp.compile(router_program, trainset=trainset, max_labeled_demos=0, max_bootstrapped_demos=0)


