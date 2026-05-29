from google.adk.models.llm_request import LlmRequest
from google.genai import types

from toolcall_rl.agents.hf_tool_call_model import (
    DEMO_TOOL_NAMES,
    _latest_function_response,
    _latest_user_text,
    _parse_tool_call,
)


def test_parse_tool_call_accepts_demo_tool_json() -> None:
    parsed = _parse_tool_call(
        '{"tool": "calculator", "args": {"expression": "94 / 2 + 11"}}'
    )

    assert parsed == {
        "tool": "calculator",
        "args": {"expression": "94 / 2 + 11"},
    }


def test_parse_tool_call_rejects_unknown_tool() -> None:
    assert _parse_tool_call('{"tool": "book_flight", "args": {"origin": "SFO"}}') is None


def test_demo_tool_set_matches_adk_agent_tools() -> None:
    assert DEMO_TOOL_NAMES == [
        "calculator",
        "google_search",
        "unit_converter",
        "text_stats",
        "string_formatter",
    ]


def test_latest_user_text_reads_last_user_message() -> None:
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="first request")],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="response")],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="latest request")],
            ),
        ]
    )

    assert _latest_user_text(request) == "latest request"


def test_latest_function_response_reads_adk_tool_result() -> None:
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Calculate 2 + 3.")],
            ),
            types.Content(
                role="function",
                parts=[
                    types.Part.from_function_response(
                        name="calculator",
                        response={"result": 5},
                    )
                ],
            ),
        ]
    )

    function_response = _latest_function_response(request)

    assert function_response is not None
    assert function_response.name == "calculator"
    assert function_response.response == {"result": 5}
