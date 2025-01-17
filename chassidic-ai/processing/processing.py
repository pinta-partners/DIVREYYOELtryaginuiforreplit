from datetime import datetime
import logging
import uuid

from litellm import aembedding
from ..retrieval.relevance.initial_relevance import (
    RelevanceChecker,
)
from ..retrieval.datasources.other_texts_datasource import OtherTextsDatasource
from ..retrieval.datasources.vector_datasource import (
    EmbeddingsDatasource,
    VectorSearchResult,
)

from ..models.models import (
    CandidatePassage,
    QueryRequest,
    QueryResponse,
    RelevanceJudgement,
)

logger = logging.getLogger(__name__)


class SearchHandler:
    def __init__(
        self,
        embedding_datasource: EmbeddingsDatasource,
        other_texts_datasource: OtherTextsDatasource,
    ):
        self.vector_datasource = embedding_datasource
        self.other_texts_datasource = other_texts_datasource
        self.relevance_checker = RelevanceChecker(
            max_parallelism=10, max_block_tokens_size=16000
        )

    async def search(self, request: QueryRequest, top_k: int = 20) -> QueryResponse:
        try:
            # TODO: Move this to a single source of truth for how to calculate at indexing and at use
            embedding_resp = await aembedding(
                model="openai/text-embedding-3-large",
                input=request.query,
                dimensions=1024,
            )
            query_vector = embedding_resp.data[0]["embedding"]
            vector_results: list[VectorSearchResult] = (
                await self.vector_datasource.find_by_knn_heb_text_openai_par(
                    query_vector=query_vector, k=top_k
                )
            )

            # Fetch full text data for results
            # TODO: Consider the dataset field in the VectorSearchResult - different search targets
            initial_passages: list[CandidatePassage] = []
            stable_ids = [result.doc.stable_id_in_ds for result in vector_results]
            # print("=======" * 10)
            # print(stable_ids)
            # print("-------" * 10)
            texts_by_ids = await self.other_texts_datasource.find_texts_by_compound_ids(
                compound_ids=stable_ids
            )
            # print(len(texts_by_ids))
            # print("=======" * 10)

            stable_id_to_text = {text.compound_id: text for text in texts_by_ids}
            for result in vector_results:
                full_text = stable_id_to_text[result.doc.stable_id_in_ds]

                initial_passages.append(
                    CandidatePassage(
                        dataset="other_books.texts_from_csv",
                        compound_id=full_text.compound_id,
                        book_name=full_text.book_name,
                        section=full_text.section,
                        topic=full_text.topic,
                        torah_num=full_text.torah_num,
                        passage_num=full_text.passage_num,
                        hebrew_text=full_text.hebrew_text,
                        en_translation=full_text.en_translation,
                        he_summary=full_text.he_summary,
                        en_summary=full_text.en_summary,
                        keywords=full_text.keywords,
                        relevance_judgments=[
                            RelevanceJudgement(
                                passage_id=full_text.compound_id,
                                model="vector_search",
                                score=float(result.score),
                                reason="",
                                timestamp=datetime.utcnow().isoformat(),
                                other_params="",
                            )
                        ],
                    )
                )

            # Add relevance judgement records
            # Step 2: pass through relevance checker
            relevance_results = await self.relevance_checker.check_relevance(
                query=request.query,
                passages=initial_passages,
            )

            # Format response
            response = QueryResponse(
                uuid=str(uuid.uuid4()),
                question=request.query,
                run_parameters={"top_k": top_k, "similarity_threshold": 0.7},
                rewrite_result={
                    "original_query": request.query,
                },
                hybrid_params={"vector_search": True, "relevance_filtering": True},
                timestamp=datetime.utcnow().isoformat(),
                passages=relevance_results,
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
