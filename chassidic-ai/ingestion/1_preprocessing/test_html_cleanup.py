import pytest
from bs4 import BeautifulSoup, NavigableString
from typing import Optional, Union


def unify_consecutive_b_elements(
    body: Union[BeautifulSoup, None]
) -> Optional[BeautifulSoup]:
    """
    Unify consecutive <b> tags if only whitespace (or no text at all) separates them.

    Args:
        body: BeautifulSoup object containing HTML elements

    Returns:
        BeautifulSoup object with unified <b> tags or None if invalid input
    """
    if not body or not isinstance(body, BeautifulSoup):
        return None

    soup = BeautifulSoup(str(body), "html.parser")
    new_content = []
    unified_b_text = []

    def flush_b_text():
        if unified_b_text:
            merged_b = soup.new_tag("b")
            merged_b.string = " ".join(filter(None, unified_b_text))
            if merged_b.string.strip():
                new_content.append(merged_b)
            unified_b_text.clear()

    for element in soup.children:
        if element.name == "b":
            # Handle nested elements inside <b>
            text = element.get_text(strip=False, separator=" ")
            unified_b_text.append(text)
        elif isinstance(element, NavigableString):
            if element.strip():
                # Non-whitespace text - flush and append
                flush_b_text()
                new_content.append(element)
            else:
                # Preserve a single space between unified elements
                if unified_b_text:
                    unified_b_text.append(" ")
        else:
            # Handle other elements
            flush_b_text()
            new_content.append(element)

    flush_b_text()

    # Clear and rebuild the body
    soup.clear()
    for item in new_content:
        soup.append(item)

    return soup


def test_unify_b_tags():
    test_cases = [
        # Test case 1: Basic adjacent b tags
        {
            "input": "<div><b>hello</b> <b>world</b></div>",
            "expected": "<div><b>hello world</b></div>",
        },
        # Test case 2: Multiple whitespace
        {
            "input": "<div><b>hello</b>    <b>world</b></div>",
            "expected": "<div><b>hello world</b></div>",
        },
        # Test case 3: Mixed content
        {
            "input": "<div>text <b>hello</b> <b>world</b> more</div>",
            "expected": "<div>text <b>hello world</b> more</div>",
        },
        # Test case 4: Nested elements
        {
            "input": "<div><b>hello <i>test</i></b> <b>world</b></div>",
            "expected": "<div><b>hello test world</b></div>",
        },
    ]

    for case in test_cases:
        input_soup = BeautifulSoup(case["input"], "html.parser")
        expected_soup = BeautifulSoup(case["expected"], "html.parser")
        result = unify_consecutive_b_elements(input_soup)

        assert str(result.strip()) == str(
            expected_soup.strip()
        ), f"Failed for input: {case['input']}"


def test_invalid_inputs():
    assert unify_consecutive_b_elements(None) is None
    assert unify_consecutive_b_elements(BeautifulSoup("", "html.parser")) is not None
    assert (
        unify_consecutive_b_elements(BeautifulSoup("<invalid>", "html.parser"))
        is not None
    )


if __name__ == "__main__":
    pytest.main([__file__])
