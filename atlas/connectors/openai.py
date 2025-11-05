"""Deprecated compatibility shim. Use litellm adapter instead."""

import warnings

from .litellm import LitellmAdapter
from atlas.config.models import AdapterType, LitellmAdapterConfig, OpenAIAdapterConfig


# Create wrapper class that warns on instantiation, not import
class OpenAIAdapter(LitellmAdapter):
    """Deprecated: Use LitellmAdapter instead."""

    def __init__(self, config):
        warnings.warn(
            "OpenAIAdapter is deprecated. Use LitellmAdapter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Convert OpenAIAdapterConfig to LitellmAdapterConfig if needed
        if isinstance(config, OpenAIAdapterConfig):
            litellm_config = LitellmAdapterConfig(
                type=AdapterType.LITELLM,
                name=config.name,
                system_prompt=config.system_prompt,
                tools=config.tools,
                llm=config.llm,
                response_format=config.response_format,
                metadata_digest=config.metadata_digest,
            )
            super().__init__(litellm_config)
        else:
            # Already a LitellmAdapterConfig or compatible
            super().__init__(config)


__all__ = ["OpenAIAdapter"]
