"""The extension against a stand-in node (pointman-extension's pointman_extension_testing), started the way a
node starts it."""

from pathlib import Path

import pytest
from pointman_extension_testing import ExtensionFailed, StandInNode

ROOT = Path(__file__).resolve().parents[1]


def test_greet_writes_the_greeting():
    with StandInNode(ROOT, settings={"greeting": "Hi"}) as node:
        done = node.run("hello.greet", {"name": "James"})
        assert done.result == {"text": "Hi, James!"}
        assert Path(done.outputs[0].path).read_text() == "Hi, James!\n"
        assert node.events["hello.last"] == {"name": "James"}


def test_the_default_greeting():
    with StandInNode(ROOT) as node:
        assert node.run("hello.greet", {"name": "Ada"}).result == {"text": "Hello, Ada!"}


def test_a_missing_name_fails():
    with StandInNode(ROOT) as node, pytest.raises(ExtensionFailed):
        node.run("hello.greet", {})
