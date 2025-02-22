import re

from pkg_resources import resource_exists, resource_filename, resource_listdir

from gymnasium_pomdps.envs.pomdp import POMDP
from gymnasium_pomdps.envs.registration import env_list, register
from gymnasium_pomdps.rendering import get_render_function
from gymnasium_pomdps.rendering.renderer import PltRenderer

__version__ = "1.0.0"

extension = "pomdp"


def list_pomdps():
    return list(env_list)


def is_pomdp(filename):  # pylint: disable=redefined-outer-name
    return filename.casefold().endswith(f".{extension.casefold()}") and resource_exists(
        "gymnasium_pomdps.pomdps", filename
    )


for filename in (
    filename
    for filename in resource_listdir("gymnasium_pomdps", "pomdps")
    if filename.casefold().endswith(f".{extension.casefold()}")
):
    path = resource_filename("gymnasium_pomdps.pomdps", filename)
    name, _ = filename.rsplit(".", 1)  # remove .pomdp extension
    version = 0

    # extract version if any
    m = re.fullmatch(r"(?P<name>.*)\.v(?P<version>\d+)", name)
    if m is not None:
        name, version = m["name"], int(m["version"])

    with open(path) as f:
        text = f.read()

    render_function = get_render_function(name)

    renderer = None if render_function is None else PltRenderer(render_function)
    register(
        id=f"POMDP-{name}-continuing-v{version}",
        entry_point="gymnasium_pomdps.envs.pomdp:POMDP",
        kwargs=dict(text=text, episodic=False, renderer=renderer),
    )

    renderer = None if render_function is None else PltRenderer(render_function)
    register(
        id=f"POMDP-{name}-episodic-v{version}",
        entry_point="gymnasium_pomdps.envs.pomdp:POMDP",
        kwargs=dict(text=text, episodic=True, renderer=renderer),
    )
