from atlas import (
    AdapterError,
    AgentAdapter,
    AtlasRewardBreakdown,
    AtlasSessionTrace,
    AtlasStepTrace,
    Evaluator,
    ExportRequest,
    ExportSummary,
    Student,
    StudentStepResult,
    Teacher,
    arun,
    build_adapter,
    create_adapter,
    create_from_atlas_config,
    export_sessions_sync,
    register_adapter,
    run,
)


def test_public_api_surface_is_importable():
    # Runtime entry points
    assert callable(run)
    assert callable(arun)

    # Personas
    assert issubclass(Student, object)
    assert issubclass(Teacher, object)
    assert issubclass(StudentStepResult, object)

    # Connector helpers
    assert issubclass(AgentAdapter, object)
    assert issubclass(AdapterError, Exception)
    for helper in (create_adapter, create_from_atlas_config, build_adapter, register_adapter):
        assert callable(helper)

    # Runtime schema types
    for schema in (AtlasRewardBreakdown, AtlasSessionTrace, AtlasStepTrace):
        assert issubclass(schema, object)

    # Evaluation facade
    assert issubclass(Evaluator, object)

    # Exporter utilities
    assert issubclass(ExportRequest, object)
    assert issubclass(ExportSummary, object)
    assert callable(export_sessions_sync)
