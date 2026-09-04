from pathlib import Path

def replace(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected text not found in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")

replace(
    "tests/test_recorder.py",
    '''    assert marks == [
        "# GAP frames=unknown t_mono_from=10.001500 t_mono_to=20.000000"
    ]
''',
    '''    assert marks == ["# GAP frames=unknown t_mono_from=10.001500 t_mono_to=20.000000"]
''',
)
replace(
    "tests/test_session.py",
    '''        assert wait_until(
            lambda: stand.session.stream_recovery_outcome
            is StreamRecoveryOutcome.BLOCKED_CONFIG
        )
''',
    '''        assert wait_until(
            lambda: stand.session.stream_recovery_outcome is StreamRecoveryOutcome.BLOCKED_CONFIG
        )
''',
)
replace(
    "tests/test_ui_widgets.py",
    '''    panel.refresh(models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=SessionState.IDLE))
''',
    '''    panel.refresh(
        models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=SessionState.IDLE)
    )
''',
)
replace(
    "tests/test_session.py",
    '''        recovery_commands = stand.sim.seen()[commands_before:]
''',
    '''        assert wait_until(lambda: len(stand.sim.seen()) >= commands_before + 5)
        recovery_commands = stand.sim.seen()[commands_before:]
''',
)
