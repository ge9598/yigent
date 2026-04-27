"""Tests for src/core/config.py field validators (audit Top10 #5 / C2 / C3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import AgentSection, PermissionsSection


class TestAgentSectionValidators:
    def test_max_iterations_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_iterations must be > 0"):
            AgentSection(max_iterations=0)

    def test_max_iterations_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_iterations must be > 0"):
            AgentSection(max_iterations=-1)

    def test_nudge_interval_zero_rejected(self) -> None:
        # nudge_interval=0 → ZeroDivisionError in bucket math (count // interval)
        with pytest.raises(ValidationError, match="nudge_interval must be > 0"):
            AgentSection(nudge_interval=0)

    def test_fork_budget_allocation_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="fork_budget_allocation must be > 0"):
            AgentSection(fork_budget_allocation=0)

    def test_positive_values_accepted(self) -> None:
        section = AgentSection(max_iterations=100, nudge_interval=20, fork_budget_allocation=30)
        assert section.max_iterations == 100
        assert section.nudge_interval == 20
        assert section.fork_budget_allocation == 30

    def test_defaults_pass_validation(self) -> None:
        section = AgentSection()
        assert section.max_iterations > 0
        assert section.nudge_interval > 0
        assert section.fork_budget_allocation > 0


class TestPermissionsSectionValidators:
    @pytest.mark.parametrize("dangerous_tool", [
        "bash", "python_repl", "write_file", "edit_file", "delete_file",
    ])
    def test_dangerous_tool_in_auto_allow_rejected(self, dangerous_tool: str) -> None:
        with pytest.raises(ValidationError, match="must not contain write/execute tools"):
            PermissionsSection(auto_allow=["read_file", dangerous_tool])

    def test_multiple_dangerous_tools_listed_in_error(self) -> None:
        with pytest.raises(ValidationError, match="bash, write_file"):
            PermissionsSection(auto_allow=["read_file", "bash", "write_file"])

    def test_safe_auto_allow_accepted(self) -> None:
        section = PermissionsSection(
            auto_allow=["read_file", "list_dir", "search_files", "web_search", "tool_search"]
        )
        assert "bash" not in section.auto_allow

    def test_default_auto_allow_passes_validation(self) -> None:
        section = PermissionsSection()
        # Defaults must not contain any of the never-auto-allow tools
        assert not (set(section.auto_allow) & PermissionsSection._NEVER_AUTO_ALLOW)

    def test_empty_auto_allow_accepted(self) -> None:
        # Maximally restrictive config — every tool prompts the user
        section = PermissionsSection(auto_allow=[])
        assert section.auto_allow == []


# ---------------------------------------------------------------------------
# Secret masking in repr (audit A8 / Top10 #15)
# ---------------------------------------------------------------------------


class TestSecretMasking:
    def test_provider_config_api_key_not_in_repr(self) -> None:
        from src.core.config import ProviderConfig
        cfg = ProviderConfig(api_key="sk-abc-secret-xyz")
        assert "sk-abc-secret-xyz" not in repr(cfg)
        assert "api_key" not in repr(cfg)  # field is fully hidden

    def test_provider_config_keys_not_in_repr(self) -> None:
        from src.core.config import ProviderConfig
        cfg = ProviderConfig(keys=["k-aaa", "k-bbb"])
        assert "k-aaa" not in repr(cfg)
        assert "k-bbb" not in repr(cfg)

    def test_provider_section_secrets_not_in_repr(self) -> None:
        from src.core.config import ProviderSection
        cfg = ProviderSection(api_key="primary-secret", keys=["k1", "k2"])
        assert "primary-secret" not in repr(cfg)
        assert "k1" not in repr(cfg)

    def test_search_tavily_key_not_in_repr(self) -> None:
        from src.core.config import SearchSection
        cfg = SearchSection(tavily_api_key="tvly-secret")
        assert "tvly-secret" not in repr(cfg)

    def test_mcp_headers_and_env_not_in_repr(self) -> None:
        from src.core.config import MCPServerConfig
        cfg = MCPServerConfig(
            name="srv",
            command=["cmd"],
            headers={"Authorization": "Bearer top-secret"},
            env={"DB_PASSWORD": "pw"},
        )
        assert "Bearer top-secret" not in repr(cfg)
        assert "pw" not in repr(cfg)

    def test_secrets_still_accessible_via_attribute(self) -> None:
        """repr=False hides from repr() but the field must still be readable."""
        from src.core.config import ProviderConfig
        cfg = ProviderConfig(api_key="sk-xyz")
        assert cfg.api_key == "sk-xyz"  # not affected by repr=False
