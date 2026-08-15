from .local_renderer import view_scene_interactive, render_360_orbit_video


def render_side_by_side(*args, **kwargs):
    from .render_side_by_side import render_side_by_side as _fn
    return _fn(*args, **kwargs)

