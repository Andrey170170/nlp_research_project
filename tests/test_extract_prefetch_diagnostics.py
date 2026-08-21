from __future__ import annotations

import json

from nlp_research_project.exact_trace_bench.extract import _summarize_artifacts


def test_artifact_summary_retains_decoder_prefetch_diagnostics(tmp_path) -> None:
    completion = tmp_path / "prompt_000" / "completion_000"
    completion.mkdir(parents=True)
    (completion / "completion.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "transcoder_diagnostics": {
                            "decoder_prefetch_request_count": 3,
                            "decoder_prefetch_load_count": 2,
                            "decoder_prefetch_load_bytes": 128,
                            "decoder_prefetch_cache_hit_count": 1,
                            "decoder_prefetch_consume_hit_count": 2,
                            "decoder_prefetch_host_wait_count": 1,
                            "decoder_prefetch_host_wait_seconds": 0.25,
                            "decoder_prefetch_in_flight_count": 0,
                            "decoder_prefetch_in_flight_high_watermark": 2,
                            "decoder_prefetch_in_flight_bytes": 0,
                            "decoder_prefetch_in_flight_bytes_high_watermark": 128,
                            "decoder_prefetch_consumer_active_count": 0,
                            "decoder_prefetch_consumer_active_bytes": 0,
                            "decoder_prefetch_consumer_retained_count": 0,
                            "decoder_prefetch_consumer_retained_bytes": 0,
                            "decoder_prefetch_consumer_retained_bytes_high_watermark": 64,
                            "decoder_prefetch_consumer_retirement_count": 2,
                            "decoder_prefetch_consumer_backpressure_count": 1,
                            "decoder_prefetch_consumer_backpressure_seconds": 0.5,
                            "decoder_prefetch_pipeline_owned_final_page_count": 0,
                            "decoder_prefetch_pipeline_owned_final_page_high_watermark": 1,
                            "decoder_prefetch_pipeline_owned_final_page_bytes": 0,
                            "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": 64,
                            "decoder_prefetch_owner_count": 0,
                            "decoder_prefetch_owner_high_watermark": 1,
                            "decoder_prefetch_owner_open_count": 2,
                            "decoder_prefetch_owner_close_count": 2,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_artifacts(tmp_path)

    assert summary["decoder_prefetch_request_count"] == 3
    assert summary["decoder_prefetch_host_wait_seconds"] == 0.25
    assert summary["decoder_prefetch_in_flight_high_watermark"] == 2
    assert summary["decoder_prefetch_in_flight_bytes_high_watermark"] == 128
    assert {
        key for key in summary if key.startswith("decoder_prefetch_")
    } == {
        "decoder_prefetch_request_count",
        "decoder_prefetch_load_count",
        "decoder_prefetch_load_bytes",
        "decoder_prefetch_cache_hit_count",
        "decoder_prefetch_consume_hit_count",
        "decoder_prefetch_host_wait_count",
        "decoder_prefetch_host_wait_seconds",
        "decoder_prefetch_in_flight_count",
        "decoder_prefetch_in_flight_high_watermark",
        "decoder_prefetch_in_flight_bytes",
        "decoder_prefetch_in_flight_bytes_high_watermark",
        "decoder_prefetch_consumer_active_count",
        "decoder_prefetch_consumer_active_bytes",
        "decoder_prefetch_consumer_retained_count",
        "decoder_prefetch_consumer_retained_bytes",
        "decoder_prefetch_consumer_retained_bytes_high_watermark",
        "decoder_prefetch_consumer_retirement_count",
        "decoder_prefetch_consumer_backpressure_count",
        "decoder_prefetch_consumer_backpressure_seconds",
        "decoder_prefetch_pipeline_owned_final_page_count",
        "decoder_prefetch_pipeline_owned_final_page_high_watermark",
        "decoder_prefetch_pipeline_owned_final_page_bytes",
        "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark",
        "decoder_prefetch_owner_count",
        "decoder_prefetch_owner_high_watermark",
        "decoder_prefetch_owner_open_count",
        "decoder_prefetch_owner_close_count",
    }
