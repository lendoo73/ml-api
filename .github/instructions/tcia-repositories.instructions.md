---
applyTo: "**/tcia/**/repositories/**/*.py"
description: "TCIA repository-layer guidance for Cancer Imaging Archive API access and persistence."
---

# TCIA Repository Instructions

- Keep TCIA data-access and external API integration logic in `tcia/**/repositories`; services and controllers should depend on repository methods rather than calling TCIA or IDC clients directly.
- Use the project's established `IDCClient` integration and repository abstractions before introducing a new HTTP or download client.
- Keep repository methods focused on one data-access concern and return domain entities or repository-defined data shapes with only the transformation needed by callers.
- Build TCIA and IDC queries from validated, normalized inputs. Parameterize values where the client supports parameters, and never interpolate untrusted request values into query text.
- Preserve TCIA identifiers and metadata accurately, including collection IDs, patient/study/series UIDs, modality, and viewer URLs.
- Surface upstream API, download, decoding, and empty-result failures explicitly with actionable errors; do not silently return fabricated or incomplete scan data.
- Keep temporary DICOM downloads and other external resources scoped to the operation and clean them up reliably.
- Cache only stable, expensive client initialization or metadata lookups using the repository's existing caching conventions; do not cache request-specific scan results without an explicit freshness policy.

## Documentation Source

- TCIA Collection Manager REST API: https://www.cancerimagingarchive.net/collection-manager-rest-api/
