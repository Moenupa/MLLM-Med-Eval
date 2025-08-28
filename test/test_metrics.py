from mmmbench.metrics import (
    BaseScore,
    GPTMetric,
    ScoreBinary,
    get_header,
    get_prompt,
)


def test_get_prompt_succ():
    for v in GPTMetric:
        assert get_prompt(v)


def test_get_prompt_fail():
    try:
        # this should fail intentionally
        get_prompt("some random string not in GPTMetric")
    except TypeError:
        pass
    else:
        assert False


def test_get_header_succ():
    for v in BaseScore.__subclasses__():
        assert get_header(v)
    assert get_header(ScoreBinary)
    print(get_header(ScoreBinary))


def test_get_header_fail():
    try:
        get_header(BaseScore)
    except TypeError:
        pass
    else:
        assert False


if __name__ == "__main__":
    test_get_header_succ()
    test_get_header_fail()
