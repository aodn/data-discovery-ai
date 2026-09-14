# unit test for model/descriptionFormattingAgent.py
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock, call
from data_discovery_ai.agents.descriptionFormattingAgent import (
    DescriptionFormattingAgent,
    chunk_text,
    format_chunk_async,
    manual_wrapper_description,
    retrieve_json,
)
from data_discovery_ai.enum.agent_enums import LlmModels


class TestDescriptionFormattingAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DescriptionFormattingAgent()
        self.agent.set_required_fields(["title", "abstract"])
        self.valid_request = {
            "title": "Test",
            "abstract": "This is a sentence.\n" * 200,
        }
        self.invalid_request = {"title": "Test", "abstract": "Short abstract"}
        self.invalid_request_missing_field = {"title": "Test"}
        self.test_formatted_abstract = """This is the formatted abstract.
        {
        "formatted_abstract": "#title **Formatted abstract**"
        }"""

    def test_is_valid_request(self):
        # Test valid request
        self.assertTrue(self.agent.is_valid_request(self.valid_request))

        # Test invalid request with missing field
        self.assertFalse(
            self.agent.is_valid_request(self.invalid_request_missing_field)
        )

    def test_make_decision_valid_request(self):
        # valid request for taking action
        self.assertTrue(self.agent.make_decision(self.valid_request))

        # short abstract, no action needed
        self.assertFalse(self.agent.make_decision(self.invalid_request))

    def test_retrieve_json_valid(self):
        output = self.test_formatted_abstract
        result = retrieve_json(model=LlmModels.OLLAMA.value, output=output)
        self.assertEqual(result, "#title **Formatted abstract**")

    @patch("data_discovery_ai.agents.descriptionFormattingAgent.logger")
    def test_retrieve_json_invalid(self, mock_logger):
        test_no_json_output = "A test string with no JSON output"
        result = retrieve_json(model=LlmModels.OLLAMA.value, output=test_no_json_output)
        self.assertEqual(result, "A test string with no JSON output")
        mock_logger.error.assert_called_once_with("No JSON found in LLM response.")

    def test_chunk_text_bounds_oversized_paragraph(self):
        abstract = " ".join(f"sentence-{i}." for i in range(1200))

        chunks = chunk_text(abstract, max_length=4000)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 4000 for chunk in chunks))
        self.assertEqual(
            " ".join(abstract.split()),
            " ".join(" ".join(chunks).split()),
        )

    def test_chunk_text_hard_splits_unbroken_text(self):
        chunks = chunk_text("x" * 10001, max_length=4000)

        self.assertEqual([4000, 4000, 2001], [len(chunk) for chunk in chunks])
        self.assertEqual("x" * 10001, "".join(chunks))

    def test_chunk_text_packs_short_paragraphs(self):
        abstract = "\n\n".join("word " * 40 for _ in range(50))

        chunks = chunk_text(abstract, max_length=4000)

        self.assertEqual(3, len(chunks))
        self.assertTrue(all(len(chunk) <= 4000 for chunk in chunks))

    def test_manual_wrapper_formats_url_and_email(self):
        abstract = "See https://example.com/data or email data@example.com."

        result = manual_wrapper_description(abstract)

        self.assertEqual(
            "See [https://example.com/data](https://example.com/data) or email "
            "[data@example.com](mailto:data@example.com).",
            result,
        )

    @patch("data_discovery_ai.agents.descriptionFormattingAgent.logger")
    @patch("data_discovery_ai.agents.descriptionFormattingAgent.chat")
    def test_execute_agent(self, mock_chat, mock_logger):
        self.agent.model_config = SimpleNamespace(
            model=LlmModels.OLLAMA.value,
            temperature=0,
            max_tokens=4000,
            response_key="summaries.ai:description",
        )
        fake_resp = MagicMock()
        fake_resp.message.content = (
            '{"formatted_abstract": "#title **Formatted abstract**"}'
        )
        mock_chat.return_value = fake_resp

        self.agent.execute(self.valid_request)

        self.assertEqual(
            self.agent.response["summaries.ai:description"],
            "#title **Formatted abstract**",
        )
        expected_response = {
            "summaries.ai:description": "#title **Formatted abstract**"
        }
        expected_calls = [
            call("Description is being reformatted by description_formatting agent"),
            call(
                f"description_formatting agent finished, it responses: \n {expected_response}",
            ),
        ]
        mock_logger.debug.assert_has_calls(expected_calls)
        self.assertEqual(mock_logger.debug.call_count, 2)


class TestDescriptionFormattingAgentAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = DescriptionFormattingAgent()
        self.agent.supervisor = SimpleNamespace(llm_client=object())
        self.agent.model_config = SimpleNamespace(
            model=LlmModels.GPT.value,
            temperature=0,
            max_tokens=1000,
            response_key="summaries.ai:description",
            chunk_size=6,
            request_timeout=20,
            total_timeout=1,
        )

    async def test_chunk_request_has_its_own_timeout(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"formatted_abstract": "formatted"}'
                        )
                    )
                ]
            )
        )

        result = await format_chunk_async(
            client,
            "system prompt",
            "source text",
            LlmModels.GPT.value,
            0,
            1000,
            20,
        )

        self.assertEqual("formatted", result)
        self.assertEqual(
            20,
            client.chat.completions.create.await_args.kwargs["timeout"],
        )

    async def test_chunks_are_formatted_sequentially_with_previous_tail(self):
        calls = []

        async def format_chunk(*args, **kwargs):
            chunk = args[2]
            calls.append((chunk, kwargs["previous_formatted_tail"]))
            return f"{chunk.upper()}. tail-{len(calls)}"

        with patch(
            "data_discovery_ai.agents.descriptionFormattingAgent.format_chunk_async",
            side_effect=format_chunk,
        ):
            result = await self.agent.take_action_async("aaaaa\n\nbbbbb\n\nccccc")

        self.assertEqual(
            [
                ("aaaaa", None),
                ("bbbbb", "tail-1"),
                ("ccccc", "tail-2"),
            ],
            calls,
        )
        self.assertEqual("AAAAA. tail-1\n\nBBBBB. tail-2\n\nCCCCC. tail-3", result)

    async def test_total_timeout_uses_raw_text_for_unfinished_chunks(self):
        self.agent.model_config.total_timeout = 0.01

        async def format_chunk(*args, **kwargs):
            chunk = args[2]
            if chunk == "first":
                return "FIRST"
            await asyncio.Event().wait()

        with patch(
            "data_discovery_ai.agents.descriptionFormattingAgent.format_chunk_async",
            side_effect=format_chunk,
        ):
            result = await self.agent.take_action_async("first\n\nsecond")

        self.assertEqual("FIRST\n\nsecond", result)


if __name__ == "__main__":
    unittest.main()
