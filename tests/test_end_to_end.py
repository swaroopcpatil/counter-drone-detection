from counter_drone.config import Config
from counter_drone.evaluate import Evaluator
from counter_drone.pipeline import CounterDronePipeline
from counter_drone.synthetic import SyntheticScene


def test_synthetic_scene_end_to_end():
    scene = SyntheticScene(seed=0, duration=12.0, dusk=(8.0, 11.0))
    pipe = CounterDronePipeline(Config(), scene.H_thermal_to_rgb, scene.rgb_size, scene.thermal_size)
    ev = Evaluator()
    for t, rgb, thermal, gts in scene:
        phase = "dusk" if scene.dusk[0] <= t <= scene.dusk[1] else "day"
        ev.add(pipe.process(rgb, thermal, t), gts, phase)
    s = ev.summary()
    assert s["detection_recall"] > 0.95
    assert s["track_coverage"] > 0.9
    assert s["declaration_rate"] > 0.8
    assert s["declaration_rate_by_phase"]["dusk"] > 0.8      # thermal carries tracks when RGB is blind
    assert s["drones_never_declared"] == []
    assert s["false_declarations_on_birds"] <= 5
    assert s["id_switches"] <= 2


def test_rgb_only_camera_still_declares_drones():
    """No thermal camera connected: drones must still be tracked and declared from RGB alone."""
    scene = SyntheticScene(seed=0, duration=10.0, dusk=(-1, -1))
    pipe = CounterDronePipeline(Config(), scene.H_thermal_to_rgb, scene.rgb_size, scene.thermal_size)
    ev = Evaluator()
    for t, rgb, _thermal, gts in scene:
        ev.add(pipe.process(rgb, None, t), gts)
    s = ev.summary()
    assert s["drones_never_declared"] == []
    assert s["declaration_rate"] > 0.6
