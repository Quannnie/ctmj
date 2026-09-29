"""Test suite for the ctmj app.

Split by concern so a failure names the broken layer:

    factories          shared fixtures (reference data + fitted model doubles)
    test_reference     lookup-table integrity
    test_models        ORM behaviour
    test_formdata      request validation
    test_predictor     the two-stage inference pipeline
    test_registry      artefact loading and failure handling
    test_views         HTTP behaviour, including every failure state
"""
