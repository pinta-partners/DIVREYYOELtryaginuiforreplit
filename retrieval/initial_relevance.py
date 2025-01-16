import asyncio
from pydantic import BaseModel
from retrieval.csv_datasource import Passage


class RelevanceResultWithId(BaseModel):
    passage_id: str
    score: float
    reason: str


class RelevanceResultWithPassage(BaseModel):
    passage: Passage
    score: float
    reason: str


class RelevanceResults(BaseModel):
    relevance_results: list[RelevanceResultWithId]


class PassageProposalBlock:
    def __init__(self, passages: list[Passage], text_proposal: str):
        self.passages = passages
        self.text_proposal = text_proposal


class RelevanceChecker:
    def __init__(self, max_parallelism: int, max_block_tokens_size: int):
        self.max_parallelism = max_parallelism
        self.max_block_tokens_size = max_block_tokens_size

    async def check_relevance(
        self, aclient, query: str, passages: list[Passage]
    ) -> list[RelevanceResultWithPassage]:
        # Split into blocks for parallel processing
        semaphore = asyncio.Semaphore(self.max_parallelism)
        blocks = self._passages_to_blocks(query, passages)
        print(
            f"Split into blocks, initial passages: {len(passages)} total blocks: {len(blocks)}"
        )
        tasks = [
            self._with_limit(
                semaphore=semaphore, aclient=aclient, query=query, passages_block=block
            )
            for block in blocks
        ]
        results = await asyncio.gather(*tasks)
        # Combine the list of lists into a single list
        results = [result for chunk in results for result in chunk]
        return results

    def _passage_to_text_proposal(self, passage: Passage) -> str:
        # Prints an identified of the passage in a way which can be extracted and then the summary
        identifier = f"=== {passage.get_id()} ==="
        return f"{identifier}\n{passage.summary}"

    def _passages_to_blocks(self, query: str, passages) -> list[PassageProposalBlock]:
        """
        Split the passages into blocks, such that each block contains passages
        whose summed text proposal length is less than self.max_block_tokens_size tokens
        """
        # Token budget for proposals in each block (optionally subtract query tokens)
        len_query = len(query.split())
        block_token_limit = self.max_block_tokens_size - len_query

        blocks = []
        current_passages = []
        current_token_count = 0

        passages_with_proposals = [
            (self._passage_to_text_proposal(p), p) for p in passages
        ]

        for proposal_text, passage in passages_with_proposals:
            proposal_tokens = len(proposal_text.split())
            # If adding this passage would exceed the token limit, start a new block
            if (
                current_token_count + proposal_tokens > block_token_limit
                and current_passages
            ):
                block_text = "\n\n".join(
                    self._passage_to_text_proposal(cp) for cp in current_passages
                )
                blocks.append(
                    PassageProposalBlock(
                        passages=current_passages, text_proposal=block_text
                    )
                )
                current_passages = []
                current_token_count = 0

            current_passages.append(passage)
            current_token_count += proposal_tokens

        # Add any remaining passages to a final block
        if current_passages:
            block_text = "\n".join(
                self._passage_to_text_proposal(cp) for cp in current_passages
            )
            blocks.append(
                PassageProposalBlock(
                    passages=current_passages, text_proposal=block_text
                )
            )

        return blocks

    async def _with_limit(
        self, aclient, semaphore, query, passages_block: PassageProposalBlock
    ) -> list[RelevanceResultWithPassage]:
        # Create an index from the passage id to the passage
        passage_id_to_passage = {p.get_id(): p for p in passages_block.passages}
        async with semaphore:
            system_message = (
                "You are a highly knowledgeable scholar and expert in the teachings of the Chassidic literature. "
                "Your task is to analyze the provided text to identify passages that best reflect the teachings or themes "
                "of the book in relation to the given question."
                "Passages are preceded with an identifier that includes the book name, section, topic, Torah number, and passage number."
                "The identifier is enclosed in triple equals (===) for easy extraction."
                "In your response, provide only identifiers without triple equals, scores (0-10), and explanations."
            )
            user_message = f"""TASK: Identify a passage from the text that provides meaningful insight into the following question:
Question: {query}
Text from the text block to analyze:
{passages_block.text_proposal}
"""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            response = await aclient.chat.completions.create(
                response_model=RelevanceResults,
                model="gpt-4o-mini",
                messages=messages,
            )

            # Match the passage to fill in RelevanceResultWithPassage
            results = []
            for result in response.relevance_results:
                passage_id = result.passage_id
                score = result.score
                reason = result.reason
                passage = passage_id_to_passage.get(passage_id)
                if passage:
                    results.append(
                        RelevanceResultWithPassage(
                            passage=passage, score=score, reason=reason
                        )
                    )
                else:
                    print(f"Error: Passage not found for id: {passage_id}")

            return results
