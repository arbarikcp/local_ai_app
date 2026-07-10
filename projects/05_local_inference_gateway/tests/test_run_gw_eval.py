import pytest

from run_gw_eval import SCENARIOS, run_eval


@pytest.mark.asyncio
class TestRunEval:
    async def test_every_scenario_passes(self):
        results = await run_eval()

        assert len(results) == len(SCENARIOS)
        failed = [r for r in results if not r.passed]
        assert failed == [], f"scenarios failed: {[(r.requirement, r.detail) for r in failed]}"

    async def test_covers_every_curriculum_requirement_by_number(self):
        results = await run_eval()
        requirement_numbers = {r.requirement.split(".")[0].rstrip("b") for r in results}
        assert requirement_numbers == {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
