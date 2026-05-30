from toolcall_rl.agents.routed_hf_tool_call_model import should_route_to_tool_model


def test_router_sends_tool_like_prompts_to_ft_model() -> None:
    assert should_route_to_tool_model("Calculate (94 / 2) + 11.")
    assert should_route_to_tool_model("Convert 10 kilometers to miles.")
    assert should_route_to_tool_model("Search Google for reward shaping.")
    assert should_route_to_tool_model('Apply titlecase to "reward driven tools".')
    assert should_route_to_tool_model('Count words in "tool calls work".')


def test_router_keeps_basic_chat_out_of_ft_model() -> None:
    assert not should_route_to_tool_model("hi")
    assert not should_route_to_tool_model("hello, how are you?")
    assert not should_route_to_tool_model("tell me something interesting")
