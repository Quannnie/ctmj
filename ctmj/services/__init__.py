"""Domain services for CJPS.

Keeping the HTTP layer thin is deliberate: ``views`` only translates between
requests and these services, so the machine-learning pipeline can be unit
tested without spinning up Django's test client.

Modules
-------
``reference_data``  Authoritative lookup tables (used by the data migration).
``registry``        Thread-safe, failure-tolerant loader for model artefacts.
``predictor``       The two-stage inference pipeline.
``formdata``        Request parsing and validation.
"""

__all__ = ["formdata", "predictor", "reference_data", "registry"]
