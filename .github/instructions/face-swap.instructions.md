---
applyTo: "face_swap/**/*.py,tests/test_face_swap*.py,README.md"
description: "Face-swap model naming and local-versus-hosted API conventions."
---

# Face-Swap Instructions

- Make each face-swap model identifier clearly communicate where prediction runs. Append `-local` to every model that performs inference in this deployment, including models running in a separate local container.
- Hosted models must identify their hosted provider or service and must not be labeled `-local`.
- Keep image/webcam and video model selectors separate when they use different backends. Expose each selector as a typed enum so the available choices appear in OpenAPI/Swagger and reject unknown values.
- When adding or renaming a face-swap model, update its enum, endpoint documentation, README examples, and integration tests together.
- Current video prediction uses the local FaceFusion worker; its model identifier is `facefusion-3.9.2-local`.
