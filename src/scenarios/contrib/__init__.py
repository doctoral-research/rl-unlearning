"""Optional domain adapters.

Everything here plugs into the language through `register_view_provider`
and is imported only on demand, so the core carries no dependency on any
particular simulator. This package is the demonstration that a new domain
family costs an adapter, not a change to the language.
"""
