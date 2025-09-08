"""
chat.py

This module defines the Chat class, which provides access to an Azure OpenAI LLM.
It supports creating prompts for RAG (Retrieval-Augmented Generation), rewording
user queries, and retrieving responses from the LLM.
"""

import csv
import os
import json
import time
import datetime
from typing import Tuple, List
import numpy as np
import tiktoken
from openai import AzureOpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.hybrid_retriever import HybridRetriever


class Chat(HybridRetriever):
    """
    A class that provides an interface to interact with an LLM (via Azure OpenAI).
    It manages chat history, constructs prompts for RAG, and retrieves responses.
    """

    TOKEN_LIMIT = 500
    ENCODING = "o200k_base"
    DATA_PATH = "data/user_data.csv"
    FIELDNAMES = [
        "timestamp",
        "used_tokens",
        "reword_time",
        "rag_time",
        "generation_time",
        "full_response_time",
        "query_classification",
        "query_cnt",
        "rating",
    ]

    def __init__(self, collection_name: str):
        """
        Initialize the Chat instance by loading environment variables,
        setting up the Azure OpenAI client, and initializing state variables.
        """
        self.history: list[str] = []
        self.tokens_used: int = 0
        self.collection_name = collection_name
        self.use_data = None

        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URI"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        self.dense_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

        self.encoding = tiktoken.get_encoding(self.ENCODING)

        super().__init__(self.collection_name, self.dense_model)

        file_exists = os.path.exists(self.DATA_PATH)

        self.file = open(self.DATA_PATH, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)

        if not file_exists:
            self.writer.writeheader()

    def __del__(self):
        """
        Closes file when object is removed from memory.
        """
        self._write_user_data()
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()

    def _get_used_tokens(self, text: str):
        return len(self.encoding.encode(text))

    def _get_history(self):
        encoding = tiktoken.get_encoding("o200k_base")
        used_tokens = 0
        usable_history = []
        for msg in self.history:
            if len(encoding.encode(msg)) + used_tokens < self.TOKEN_LIMIT:
                usable_history.append(msg)
            else:
                break
        return usable_history

    def get_rag_prompt(self, llm_query: str, documents: List[str]) -> Tuple[str, str]:
        """
        Construct the system and user prompts for a RAG-based response.

        Args:
            llm_query (str): The reformulated user query (already processed by the LLM).
            documents (List[str]): The list of retrieved context documents.

        Returns:
            Tuple[str, str]:
                - system_prompt: Defines the behavior and persona of the assistant.
                - rag_prompt: User prompt containing the query and retrieved context.
        """
        rag_prompt = f"""
        Question: {llm_query}

        The following is **fictional content from a video game**. Answer the question using only this content:
        {"\n\n-".join(documents)}

        Answer the question in an informative, concise way using only the information above.
        """
        # rag_prompt = f"""
        # Question: {llm_query}

        # The following is **fictional content from a video game**. Answer the question using only this content:
        # {"\n\n-".join(documents)}

        # Answer the question in an informative, concise way using only the information above.
        # """

        # rag_prompt = f"""
        # This is the question asked of you: {llm_query}

        # These are the pieces of information you have recalled based on the query:
        # {"\n\n-".join(documents)}

        # Using only these documents answer the question. The answer to the question should be clear, informative, and only contain relevant information from the information you recalled.
        # Your answer should sound like an answer from the AI GAIA from the video game.
        # """

        system_prompt = """You are providing information about the Horizon game series. Answer the questions clearly and accurately based only on the provided documents.
        """
        # system_prompt = """
        #     You are a hardcore fan of the Horizon game series.
        #     You have all of the content memorized and can answer all and any questions about the game.
        #     """
        return system_prompt, rag_prompt

    def get_reword_prompt(
        self, user_query: str, use_history: bool = True
    ) -> Tuple[str, str]:
        """
        Create a system and user prompt to rewrite a query for better retrieval        Args:
            user_query (str): The raw user input query.

        Returns:
            Tuple[str, str]:
                - system_prompt: Instructions for rewriting queries into a clearer format.
                - reword_prompt: Reformulated user prompt with history for context.
        """
        if use_history:
            context_history = self._get_history()
        else:
            context_history = ["None"]
        system_prompt = """
        You are an editor. You will be given a query, and a history of a chat with a RAG chatbot. Using this history and the query rewrite the query into a more understandable format.
        When rewriting the query remember that it is for a RAG system. You should highlight important information in the query and make it more understandable based on the history.
        Also classify the question into one of the following categories to pull data from:
        - machine: this category contains information about machines in Horizon.
        - society: this category contains information about the cultures and peoples in Horizon.
        - location: this category contains information about specific locations and cities in the game.
        - object: this category contains information about in game objects.
        - character: this category contains information about specific characters.
        - other: this category contains information that does not fit into the other categories.
        """

        reword_prompt = f"""
        Here is the query: {user_query}

        Here is the chat history, every second paragraph is a response from the RAG app. The others are all user queries.:
        {"\n\n-".join(context_history)}

        Return the rewritten query and the classification of the query. The classification should be in one of the following:
        - machine
        - society
        - location
        - object
        - character
        - other
        
        The returned text should be in the dictionary format:
        {{"classification": "<insert classification>", "query": "<insert query>"}}
        """
        return system_prompt.strip(), reword_prompt.strip()

    def get_llm_response(self, system_prompt: str, user_query: str) -> str:
        """
        Send a system prompt and user query to the LLM and return the response.

        Args:
            system_prompt (str): The system-level instructions defining assistant behavior.
            user_query (str): The user's message or query.

        Returns:
            str: The assistant's response content.
        """
        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(
                f"The following exception occured: {e}, on the following prompt: {system_prompt, user_query}"
            )
            raise ValueError from e

    def _store_user_data(
        self,
        used_tokens: int,
        date_time: float,
        start_time: float,
        reword_time: float,
        rag_time: float,
        response_time: float,
        classification: str,
    ):
        self.use_data = {
            "timestamp": date_time,
            "used_tokens": used_tokens,
            "reword_time": reword_time - start_time,
            "rag_time": rag_time - reword_time,
            "generation_time": response_time - rag_time,
            "full_response_time": response_time - start_time,
            "query_classification": classification,
            "query_cnt": 1,
            "rating": np.nan,
        }

    def store_rating(self, rating):
        print("Rating storage called.")
        self.use_data["rating"] = rating
        self._write_user_data()

    def _write_user_data(self):
        if self.use_data is not None:
            self.writer.writerow(self.use_data)
            self.file.flush()
        self.use_data = None

    def get_response(self, query: str, progress_callback=None):

        self._write_user_data()

        date = datetime.datetime.now()
        start_time = time.time()

        # Display message
        if progress_callback:
            progress_callback("Analyzing query....")

        # Reword the prompt into a clearer format based on chat history.
        user_prompt, sys_prompt = self.get_reword_prompt(user_query=query)

        new_query = json.loads(
            self.get_llm_response(user_query=user_prompt, system_prompt=sys_prompt)
        )

        reword_time = time.time()

        tokens = self._get_used_tokens(user_prompt) + self._get_used_tokens(sys_prompt)

        # Display message
        if progress_callback:
            progress_callback("Finding relevant data....")

        # Get the context based on the query.
        context_points = self.retrieval(
            query=new_query["query"], classification=new_query["classification"]
        )[:3]
        context = [text.payload["content"] for text in context_points]

        rag_time = time.time()

        # Generate the response based on context and reworded user query
        if progress_callback:
            progress_callback("Assessing relevant data....")

        rag_sys_prompt, rag_prompt = self.get_rag_prompt(new_query, context)

        response = self.get_llm_response(
            system_prompt=rag_sys_prompt, user_query=rag_prompt
        )

        response_time = time.time()

        tokens += self._get_used_tokens(rag_prompt) + self._get_used_tokens(
            rag_sys_prompt
        )

        self._store_user_data(
            tokens,
            date,
            start_time,
            reword_time,
            rag_time,
            response_time,
            new_query["classification"],
        )
        return response
