"""End-to-end physics tests: does the simulated robot actually finish the job?

Each test runs the full simulation without rendering, so it takes a few seconds.
"""

from sortbot import COLORS, Controller, Sim, Skills, execute, parse_command


def make_robot(seed=None):
    sim = Sim()
    if seed is not None:
        sim.reset(sim.random_layout(seed))
    skills = Skills(Controller(sim), say=lambda msg: None)
    return sim, skills


def test_sort_default_layout():
    sim, skills = make_robot()
    execute(parse_command("sort the cubes by color"), skills)
    assert all(sim.cube_on_pad(c) for c in COLORS)


def test_sort_random_layouts():
    for seed in (1, 2, 3):
        sim, skills = make_robot(seed)
        assert skills.sort(), f"seed {seed} failed"


def test_tower_after_sort():
    sim, skills = make_robot()
    execute(parse_command("sort the cubes. build a tower: blue, green, red"), skills)
    assert skills.resting_on("green") == "blue"
    assert skills.resting_on("red") == "green"
    assert sim.cube_pos("red")[2] > 0.11   # third level: 2.5 + 5 + 5 cm


def test_swap_needs_a_temporary_spot():
    sim, skills = make_robot()
    skills.sort()
    execute(parse_command("put red on the blue pad and blue on the red pad"), skills)
    assert sim.cube_on_pad("red", "blue")
    assert sim.cube_on_pad("blue", "red")
