import time

from local_ai_core.tracing.trace import (
    CORE_REQUIRED_STEPS,
    TraceBuilder,
    validate_trace_shape,
)


class TestSpanMeasuresRealElapsedTime:
    def test_a_span_around_real_work_has_a_positive_elapsed_ms(self):
        builder = TraceBuilder(request_id="trace-1")
        with builder.span("model_call"):
            time.sleep(0.01)
        trace = builder.build()
        assert trace.spans[0].elapsed_ms >= 10

    def test_span_attributes_are_preserved(self):
        builder = TraceBuilder(request_id="trace-1")
        with builder.span("input_validation", schema="TicketRequest"):
            pass
        trace = builder.build()
        assert trace.spans[0].attributes == {"schema": "TicketRequest"}


class TestRecordRetrievalStep:
    def test_records_query_and_chunk_id_spans(self):
        builder = TraceBuilder(request_id="trace-2")
        builder.record_retrieval_step(query="how do I reset my password?", chunk_ids=["doc1::0", "doc1::1"])
        names = builder.build().span_names()
        assert "retrieval_query" in names
        assert "retrieved_chunk_ids" in names
        assert "reranker_scores" not in names

    def test_reranker_scores_span_is_recorded_when_provided(self):
        builder = TraceBuilder(request_id="trace-2")
        builder.record_retrieval_step(query="q", chunk_ids=["doc1::0"], reranker_scores=[0.9, 0.4])
        assert "reranker_scores" in builder.build().span_names()


class TestRecordToolCallStep:
    def test_records_a_tool_call_span_with_arguments_and_result(self):
        builder = TraceBuilder(request_id="trace-3")
        builder.record_tool_call_step(tool_name="lookup_order", arguments={"order_id": "123"}, result={"status": "shipped"})
        trace = builder.build()
        assert trace.spans[0].name == "tool_calls"
        assert trace.spans[0].attributes["tool_name"] == "lookup_order"
        assert trace.spans[0].attributes["error"] is None


class TestRecordAgentStep:
    def test_records_an_agent_step_span(self):
        builder = TraceBuilder(request_id="trace-4")
        builder.record_agent_step(step_index=0, action="search", observation="found 3 results")
        trace = builder.build()
        assert trace.spans[0].name == "agent_step"
        assert trace.spans[0].attributes["step_index"] == 0


class TestValidateTraceShape:
    def test_a_complete_core_trace_has_no_missing_steps(self):
        builder = TraceBuilder(request_id="trace-5")
        with builder.span("input_validation"):
            pass
        with builder.span("model_call"):
            pass
        with builder.span("final_response"):
            pass
        assert validate_trace_shape(builder.build()) == []

    def test_an_incomplete_trace_reports_the_missing_core_steps(self):
        builder = TraceBuilder(request_id="trace-6")
        with builder.span("input_validation"):
            pass
        missing = validate_trace_shape(builder.build())
        assert missing == ["model_call", "final_response"]

    def test_missing_optional_retrieval_steps_does_not_fail_core_validation(self):
        builder = TraceBuilder(request_id="trace-7")
        for step in CORE_REQUIRED_STEPS:
            with builder.span(step):
                pass
        assert validate_trace_shape(builder.build()) == []
