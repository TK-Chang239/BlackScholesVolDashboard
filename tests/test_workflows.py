"""The daily workflow's binding properties, asserted as data (SPEC 2.1, 4)."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / ".github" / "workflows" / "daily.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def load(path: Path) -> dict:
    # PyYAML parses the unquoted key `on:` as the boolean True, so both spellings
    # are accepted here rather than being worked around in the workflow file.
    doc = yaml.safe_load(path.read_text())
    doc["on"] = doc.get("on", doc.get(True))
    return doc


class TestDailyWorkflow:
    def test_exists_and_parses(self):
        assert DAILY.exists()
        assert load(DAILY)["jobs"]

    def test_cron_is_the_spec_schedule(self):
        triggers = load(DAILY)["on"]
        assert triggers["schedule"] == [{"cron": "17 2 * * 2-6"}]

    def test_cron_avoids_the_hour_and_half_hour_github_congests(self):
        # GitHub does not promise punctuality and delays scheduled runs most at
        # :00 and :30. Run 33477912528 was dispatched five hours after its
        # 01:30 slot, into 02:30 ET, where the book is shut and the fetch comes
        # back thin. An odd minute is not a guarantee, only a smaller target.
        minute = int(load(DAILY)["on"]["schedule"][0]["cron"].split()[0])
        assert minute % 30 != 0, "':00' and ':30' are the most congested slots"

    def test_cron_still_lands_after_the_us_close_in_both_dst_regimes(self):
        import datetime as dt
        from zoneinfo import ZoneInfo
        minute, hour = (int(f) for f in load(DAILY)["on"]["schedule"][0]["cron"].split()[:2])
        et = ZoneInfo("America/New_York")
        for summer_or_winter in (dt.date(2026, 9, 2), dt.date(2026, 1, 15)):
            fired = dt.datetime.combine(summer_or_winter, dt.time(hour, minute),
                                        tzinfo=dt.timezone.utc).astimezone(et)
            # after the 16:00 close, and before midnight rolls the book over
            assert dt.time(16, 0) < fired.time() < dt.time(23, 59), fired

    def test_manual_dispatch_is_available(self):
        assert "workflow_dispatch" in load(DAILY)["on"]

    def test_tests_run_before_the_pipeline(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        runs = [s.get("run", "") for s in steps]
        test_at = next(i for i, r in enumerate(runs) if "pytest" in r)
        run_at = next(i for i, r in enumerate(runs) if "src.run_daily" in r)
        assert test_at < run_at, "a broken model must never publish"

    def test_secrets_arrive_as_env_and_are_never_inlined(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        step = next(s for s in steps if "src.run_daily" in s.get("run", ""))
        assert set(step["env"]) == {"EODHD_API_TOKEN", "MASSIVE_API_KEY"}
        for value in step["env"].values():
            assert value.startswith("${{ secrets."), value
        assert "${{ secrets." not in "".join(s.get("run", "") for s in steps)
        # A secret wired into some other step's `env:` block would slip past the
        # two checks above (they only look at `run:` text and at this one step's
        # own `env`), so also confirm no OTHER step's env references a secret.
        for other in steps:
            if other is step:
                continue
            for value in (other.get("env") or {}).values():
                assert "${{ secrets." not in str(value), (other, value)

    def test_job_can_write_and_will_not_overlap_itself(self):
        doc = load(DAILY)
        assert doc["jobs"]["daily"]["permissions"]["contents"] == "write"
        assert doc["concurrency"]["group"]
        assert doc["concurrency"]["cancel-in-progress"] is False

    def test_job_has_a_timeout_well_short_of_the_default(self):
        # run_daily sets no request timeout, and cancel-in-progress is false, so
        # a stuck job would otherwise hold the daily-publish concurrency group
        # for GitHub's 6-hour default -- blocking every later scheduled run and
        # any manual recovery dispatch queued behind it.
        timeout = load(DAILY)["jobs"]["daily"]["timeout-minutes"]
        assert 0 < timeout < 360

    def test_commit_is_skipped_when_the_pipeline_did_not_publish(self):
        # An off-hours refusal exits 0 so the run is not a false red, but
        # `run` has already upserted the underlying bars by then, so `git add
        # data docs` would find a diff and commit a "daily snapshot" that
        # contains no snapshot. Gate the commit on the pipeline saying it
        # published rather than on the working tree being dirty.
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        pipeline = next(s for s in steps if "src.run_daily" in s.get("run", ""))
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert pipeline.get("id"), "the commit step needs a step to refer to"
        assert commit.get("if") == f"steps.{pipeline['id']}.outputs.published == 'true'"

    def test_a_failed_pipeline_still_never_reaches_the_commit(self):
        # A custom `if:` on a step is implicitly ANDed with success(), so
        # gating on the published output does not re-open the commit on a
        # failed run -- but only while the expression names no status
        # function. always()/failure() would opt it back in.
        commit = next(s for s in load(DAILY)["jobs"]["daily"]["steps"]
                      if "git commit" in s.get("run", ""))
        for escape_hatch in ("always()", "failure()", "cancelled()"):
            assert escape_hatch not in commit["if"], escape_hatch

    def test_commit_is_conditional_on_a_real_change(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "git diff --quiet" in commit["run"] or "--staged --quiet" in commit["run"]

    def test_bot_commit_does_not_retrigger_ci(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "[skip ci]" in commit["run"]

    def test_only_data_and_docs_are_committed(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "git add data docs" in commit["run"]
        assert "git add ." not in commit["run"]
        assert "git add -A" not in commit["run"]


class TestCIWorkflow:
    def test_still_runs_tests_on_push(self):
        doc = load(CI)
        assert "push" in doc["on"]
        assert any("pytest" in s.get("run", "") for s in doc["jobs"]["tests"]["steps"])

    def test_python_version_matches_the_daily_job(self):
        def python_of(doc):
            for job in doc["jobs"].values():
                for step in job["steps"]:
                    if "setup-python" in str(step.get("uses", "")):
                        return str(step["with"]["python-version"])
            return None
        assert python_of(load(CI)) == python_of(load(DAILY))
