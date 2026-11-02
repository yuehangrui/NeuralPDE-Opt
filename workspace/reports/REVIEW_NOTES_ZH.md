# Standalone edition: final review notes

2026-10-29. These notes were added after the full pipeline passed. The original automated reports, numerical sources, weights, data, and FINAL_AUDIT were retained unchanged in v1.0.0. The English edition translates presentation text and records its new hashes separately.

- Full pipeline passed with exit code 0; 156 training runs and chapter evaluations/audits completed. Final re-verification of 3364 artifact hashes passed.
- The original 9550 frozen files (7868164617 bytes) were unchanged; the independent workspace's 317 source files matched their frozen training identities.
- The original Chapter 4 summary.md claim of 9 reused seed0 runs came from an unconditional inherited template. Here adopted_jobs.json is []; frozen_jobs has 27 entries and additional_jobs has 21, and the pipeline does not pass --reuse-seed0. All 48 runs were trained from scratch. summary.reviewed.md corrects this sentence only; numerical tables match the original summary.
- INSTALLATION.json retains the earlier wheel identity from the initial installation. The formal training version is identified by PRETRAIN_SOURCE_UPDATE.json, DEPENDENCIES.json, PAYLOAD_MANIFEST.json at v1.0.0, and the run record. The original formal wheel SHA256 is f7c59bec496898be46b223f4d401b66091c9d65b9b611be0b7b834bdcf9534a9; its source-manifest SHA256 is e07d6acdeb2a9365952196f996ef8c8c039917f39a219097a8aa847dd77afef6. The English edition has distinct package/source identities.
- The 68 rows map algorithms/mechanisms, not 68 distinct leaf sections. Chapters 1 and 6 are explanatory; deferred work is not counted as complete.
- These tests have been viewed and must not guide further model selection or tuning of this version. Completing budgets is not strict optimality; all accuracy differences and nonzero residuals are retained.
- The local delivery mirrored source and 2426 original evidence files; model weights and data remained in the complete independent server workspace. The external LOCAL_SYNC_VERIFIED.json records that transfer. Selected models and final-test arrays are now also available in Release attachments.

The LaTeX/PDF report combines automated results, the complete manuscript mapping, dependency limits, original-asset protection, and the correction above. Publication artifacts have their own manifest and do not replace FINAL_AUDIT.json.
