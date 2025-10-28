# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Union

from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


def format_context(context: Union[str, Dict[str, str]]) -> str:
    if isinstance(context, str):
        return context
    else:
        context_string = ""
        while isinstance(context, dict):
            context_key = list(context.keys())[0]
            context_string += f"{context_key}: {context[context_key]}\n"
            context = context[context_key]
        return context_string


class ExperiencePool:
    """This experience pool has query/key that are dictionaries of context/question"""
    def __init__(
        self,
        correct_trajectories_path: str,
        embed_type: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_provider: str = "huggingface",
    ):
        self.embed_type = embed_type
        store = {}
        # the jsonl file is a list of dictionaries, each dictionary contains a key and a value
        correct_trajectories_dicts: List[Dict[str, Dict[str, str]]] = load_dataset('json', data_files=correct_trajectories_path)['train']
        for correct_trajectories_dict in correct_trajectories_dicts:
            key = correct_trajectories_dict['key']
            value = correct_trajectories_dict['value']
            k = self._get_embed_string(key, embed_type)
            value.update(key)
            store[k] = dict(value)
        if embedding_provider == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        elif embedding_provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        else:
            raise ValueError(f"Invalid embedding provider: {embedding_provider}")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        for embed_string, meta_data in store.items():
            self.vector_store.add_documents([Document(page_content=embed_string, metadata=meta_data)])

    @staticmethod
    def _get_embed_string(embed_dict: Dict[str, str], embed_type: str) -> str:
        context = format_context(embed_dict['context'])
        if embed_type == 'both':
            s = context + '\n' + embed_dict['question']
        elif embed_type == 'question':
            s = embed_dict['question']
        elif embed_type == 'context':
            s = context
        return s

    def get_demonstrations(self, query: Dict[str, str], k: int) -> List[Dict[str, str]]:
        q = self._get_embed_string(query, self.embed_type)
        retrieved_docs = self.vector_store.similarity_search(q, k=k)
        return self.format_demonstration([doc.metadata for doc in retrieved_docs])

    def format_demonstration(self, demonstrations: List[Dict[str, str]]) -> str:
        demo_str = demonstrations[0]['messages'][1]['content'] + '\n'
        for m in demonstrations[0]['messages'][2:]:
            if m['role'] == 'user':
                demo_str += f"Observation: {m['content'][:10000]}\n\n"
            else:
                demo_str += f"{m['content']}\n"
        return demo_str



if __name__ == "__main__":
    experience_pool = ExperiencePool(
        correct_trajectories_path='/Users/kevin/Downloads/SecRL/secgym/agents/expel_train/corrects_30.jsonl',
        embed_type='both'
    )
    print(experience_pool.get_demonstrations({'context': 'context', 'question': 'What is the capital of France?'}, 1))