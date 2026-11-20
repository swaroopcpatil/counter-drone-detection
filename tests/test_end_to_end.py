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
