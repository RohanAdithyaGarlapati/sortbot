"""Unit tests for the keyword parser. Run with:  python -m pytest"""

from sortbot.language import actions_from_plan, parse_command


def kinds(text):
    return [str(a) for a in parse_command(text)]


def test_sort():
    assert kinds("Sort the cubes by color") == ["sort()"]
    assert kinds("please tidy up") == ["sort()"]


def test_place_on_pad_and_cube():
    assert kinds("put the red cube on the blue pad") == ["place(red -> blue_pad)"]
    assert kinds("stack green on red") == ["place(green -> red_cube)"]
    assert kinds("put blue on top of the green block") == ["place(blue -> green_cube)"]


def test_sequences():
    assert kinds("put red on the green pad, then go home") == ["place(red -> green_pad)", "home()"]
    assert kinds("red on the blue pad and green on the red pad") == [
        "place(red -> blue_pad)", "place(green -> red_pad)"]


def test_towers():
    assert kinds("build a tower: blue, green, red") == ["tower(blue, green, red)"]
    assert kinds("make a tower with red at the bottom") == ["tower(red, blue, green)"]
    assert kinds("build a tower with blue at the bottom and red on top") == ["tower(blue, green, red)"]
    assert kinds("now build a tower: green, then blue, then red on top") == ["tower(green, blue, red)"]
    assert kinds("stack them") == ["tower(blue, green, red)"]


def test_nonsense_gives_nothing():
    assert kinds("make me a sandwich") == []


def test_llm_plan_validation_drops_bad_steps():
    plan = {"steps": [
        {"action": "place", "object": "red", "target": "blue_pad"},
        {"action": "place", "object": "purple", "target": "blue_pad"},   # unknown cube
        {"action": "place", "object": "red", "target": "red_table"},     # unknown target
        {"action": "tower", "order": ["red", "red"]},                    # duplicate colors
        {"action": "explode"},                                           # not a skill
        {"action": "home"},
    ]}
    assert [str(a) for a in actions_from_plan(plan)] == ["place(red -> blue_pad)", "home()"]
