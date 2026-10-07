from swarmlab.participants.llm import LLMAgent


def test_report_json_accepts_answer_key_aliases():
    agent = LLMAgent(model="fake:country_reporter", report_json=True, report_fields=["country", "reason"])
    agent.last_observation = [{"type": "text", "text": 'Allowed countries: ["Peru", "Poland"]'}]
    for text in ('{"answer": "Peru"}', '{"guess": "peru", "reason": "red and white"}', '{"flag": "Peru"}',
                 '{"country": "Peru", "reason": "stripes"}'):
        report, err = agent._parse_report(text)
        assert err is None, (text, err)
        assert report["country"] == "Peru", text


def test_report_json_single_unknown_key_falls_back_to_its_string_value():
    agent = LLMAgent(model="fake:country_reporter", report_json=True, report_fields=["country"])
    agent.last_observation = [{"type": "text", "text": 'Allowed countries: ["Peru", "Poland"]'}]
    report, err = agent._parse_report('{"nation": "Poland"}')
    assert err is None and report["country"] == "Poland"
    report, err = agent._parse_report('{"x": 1, "y": "Poland"}')  # ambiguous: two keys, not a lone string
    assert report is None and "no string" in err
