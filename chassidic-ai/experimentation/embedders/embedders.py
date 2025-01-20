# embedders.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import litellm
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM


# To consider:
# 1. voyage-3-large (via litellm) (see https://docs.voyageai.com/docs/embeddings#models-and-specifics) NOTE: input type
# 2. dicta-il/BEREL_2.0
# 3. nvidia/NV-Embed-v2 (see https://huggingface.co/nvidia/NV-Embed-v2) # Note - require instructions
# 4. openai/text-embedding-3-large (via litellm)


class BaseEmbedder(ABC):
    """
    An abstract base class defining the interface for all embedders.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Returns the embedding vector dimension."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Returns the maximum number of tokens supported by the model."""

    @abstractmethod
    async def embed_query(self, text: str, instruction: str | None) -> np.ndarray:
        """Generates an embedding for a single text."""

    @abstractmethod
    async def embed_passages(self, texts: List[str]) -> list[np.ndarray]:
        """
        Optional convenience method for batch embedding.
        Default to calling `self.embed` for each text individually.
        """


# Embedder using litellm embedders
class LitellmEmbedder(BaseEmbedder):
    """
    An embedder that uses litellm to generate embeddings.
    """

    def __init__(self, model: str, dimensions: int, max_tokens: int) -> None:
        self.model = model
        self.dimensions = dimensions
        self.max_tokens

    @property
    def dim(self) -> int:
        return self.dimensions

    @property
    def max_tokens(self) -> int:
        return self.max_tokens

    async def embed_query(
        self, text: str, instruction: str | None, **kwargs
    ) -> np.ndarray:
        # NOTE: Check each model to see if it requires instructions
        embedding_resp = await litellm.aembedding(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            **kwargs,
        )
        # Verify embedding dimensions
        assert embedding_resp.data[0]["embedding"].shape[0] == self.dimensions
        return embedding_resp.data[0]["embedding"]

    async def embed_passages(self, texts: List[str], **kwargs) -> list[np.ndarray]:
        embedding_resp = await litellm.aembedding(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            **kwargs,
        )
        # Verify embedding dimensions
        assert all(
            item["embedding"].shape[0] == self.dimensions
            for item in embedding_resp.data
        )
        return [item["embedding"] for item in embedding_resp.data]


# Embedder using BERT-based huggingface transformers (i.e. "dicta-il/BEREL_2.0")
class HuggingfaceBertBasedEmbedder(BaseEmbedder):
    """
    An embedder that uses huggingface transformers to generate embeddings.
    """

    def __init__(self, model_id: str, dimensions: int) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = BertForMaskedLM.from_pretrained(model_id)

    @property
    def dim(self) -> int:
        return self.dimensions

    @property
    def max_tokens(self) -> int:
        return self.tokenizer.model_max_length

    async def embed_query(self, text: str, instruction: str | None) -> np.ndarray:
        # NOTE: Check each model to see if it requires instructions
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        # Verify embedding dimensions
        assert outputs.last_hidden_state.shape[1] == self.dimensions
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    async def embed_passages(self, texts: List[str]) -> list[np.ndarray]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        # Verify embedding dimensions
        assert outputs.last_hidden_state.shape[1] == self.dimensions
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()


class NVEmbedBasedEmbedder(BaseEmbedder):
    """
    An embedder that uses NV-Embed-v2 to generate embeddings.
    """

    def __init__(self, model_id: str, dimensions: int = 4096) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    @property
    def dim(self) -> int:
        return self.dimensions

    @property
    def max_tokens(self) -> int:
        return self.tokenizer.model_max_length

    async def embed_query(self, text: str, instruction: str | None) -> np.ndarray:
        # See https://huggingface.co/nvidia/NV-Embed-v2
        if not instruction:
            instruction = "Given a question, retrieve passages that answer the question"
        query_prefix = f"Instruct: {instruction}\nQuery: "
        inputs = self.tokenizer(
            query_prefix + text,
            return_tensors="pt",
            max_length=self.dimensions,
            truncation=True,
        )
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        # Verify embedding dimensions
        assert embedding.shape == (self.dimensions,)
        return embedding

    async def embed_passages(self, texts: List[str]) -> List[np.ndarray]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.dimensions,
        )
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        # Verify embedding dimensions
        assert embeddings.shape == (len(texts), self.dimensions)
        return embeddings.tolist()


class VoyageEmbedder(LitellmEmbedder):
    """
    Uses the voyage-3-large model to generate embeddings.
    Provides the instructions and the additional input_type arguments.
    """

    def __init__(self, dimensions: int = 1024, max_tokens: int = 32000) -> None:
        super().__init__(
            model="voyage-3-large", dimensions=dimensions, max_tokens=max_tokens
        )

    async def embed_query(
        self, text: str, instruction: str | None = None, **kwargs
    ) -> np.ndarray:
        # Add input_type to kwargs
        kwargs["input_type"] = "query"
        if not instruction:
            instruction = "Given a question, retrieve passages that answer the question"
        return await super().embed_query(text, instruction, **kwargs)

    async def embed_passages(self, texts: List[str], **kwargs) -> list[np.ndarray]:
        # Add input_type to kwargs
        kwargs["input_type"] = "document"
        return await super().embed_passages(texts, **kwargs)


# Create specific objects for each embedder
openai_3_large_3027_embedder = LitellmEmbedder(
    model="openai/text-embedding-3-large", dimensions=3072, max_tokens=8191
)

voyage_3_large_1024_embedder = VoyageEmbedder(dimensions=1024, max_tokens=32000)

berel_2_embedder = HuggingfaceBertBasedEmbedder(
    model_id="dicta-il/BEREL_2.0", dimensions=768
)

nv_embedder = NVEmbedBasedEmbedder(
    model_id="nvidia/NV-Embed-v2", dimensions=4096  # , max_tokens=32768
)
