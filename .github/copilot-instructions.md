# Project Instructions

- For an explicit request such as **"deploy this update"**, use `.github/prompts/deploy-update.prompt.md` as the deployment checklist. Complete its repository push, Docker Hub publish, applicable Compose updates, local Cloudflare deployment, and post-deployment verification steps without requiring the user to restate them.
- Treat the prompt as authorization for those deployment actions, but preserve model-weight volumes and unrelated containers/images. Never create or move a `latest` tag unless the user explicitly requests it.
- Face-swap video inference runs in the separate `face-swap-video-worker` container; the Qwen image-generation models remain the image/webcam backends. Keep the worker's FaceFusion model assets under the shared `/app/weights/face-swap/facefusion` directory.
- For face-swap model identifiers and endpoint changes, follow `.github/instructions/face-swap.instructions.md`.

## Company References

- No relevant company-specific guidance for face-swap model naming was found in Confluence. The naming rule is project-specific and user-directed.
