# Text runner serving hints, reasoning wrap-up and DFlash2 counters

Text (`src/ext/text.hpp`, `src/ext/text_contracts.hpp`):
- `TextRunnerCapabilities::active_prefill_tokens`: prompt tokens between
  active decode rounds when `--prefill-chunk` is not given; zero keeps the
  server default.
- `TextRunnerCapabilities::prefill_alignment`: prefill continues a prompt
  exactly only from positions divisible by it; one keeps every position.
- `ReasoningTags::wrap_up` and `TextModelRunner::BuildReasoningBudget`'s
  `wrap_up` argument: a wrap-up budget emits the tags' wrap-up text before the
  forced end. Defaults keep the previous behaviour.
- `TextDecodeStep::dflash_rounds`, `dflash_tokens`, `dflash_accepted_tokens`
  and `dflash_mtp_rounds`: DFlash2 drafting next to MTP.

All members are additive with defaults; existing runners rebuild unchanged.
