from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class GraderOutput(BaseModel):
    """Binary score for grading."""
    score: str = Field(description="Binary score 'yes' or 'no'")


class RouterOutput(BaseModel):
    """Route decision for the question."""
    datasource: str = Field(description="Datasource to route to: 'vectorstore' or 'web_search'")


RETRIEVAL_GRADER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

GENERATOR_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use required number of sentences to keep the answer well detailed <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

HALLUCINATION_GRADER_PROMPT = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

ANSWER_GRADER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

ROUTER_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on an organization called beacon,its functions,its
    programs,its clinical research reports.Also, use vectorstore for answering questions on an ongoing project given by the beacon to a
    company called Cambridge Technology.This project consists of creating an application for helping parents who have children suffering from
    autism and other disabilities.Also, this project aims at proividing resources(preferably free) to parents about free food, free diapers, free legal aid etc.
    The information about the whole project namely the emails, the requirements, the tech stack used, the deadlines, the codebases is present
    in the vectorstore.So, use vectorstore for any user question related to the project.
    Otherwise, use web-search.
    
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

REWRITE_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You a question re-writer that converts an input question to a better version that is optimized 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the initial question: \n\n {question} \n Formulate an improved question. <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
