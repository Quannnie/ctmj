"""Template helpers for the prediction form.

The core problem these solve: Django's ``{{ errors.Age }}`` syntax performs a
*literal* key lookup, so ``{{ errors.name }}`` looks up the key ``"name"`` and
never the field name held in the ``name`` variable. Rendering a per-field
error from an ``{% include ... with name='Age' %}`` therefore needs a filter
that receives the key as an argument.
"""
