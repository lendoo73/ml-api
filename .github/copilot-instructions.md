# Project Instructions

- For an explicit request such as **"deploy this update"**, use `.github/prompts/deploy-update.prompt.md` as the deployment checklist. Complete its repository push, Docker Hub publish, applicable Compose updates, local Cloudflare deployment, and post-deployment verification steps without requiring the user to restate them.
- Treat the prompt as authorization for those deployment actions, but preserve model-weight volumes and unrelated containers/images. Never create or move a `latest` tag unless the user explicitly requests it.
- Face-swap video inference runs in the separate `face-swap-video-worker` container; the Qwen image-generation models remain the image/webcam backends. Keep the worker's FaceFusion model assets under the shared `/app/weights/face-swap/facefusion` directory.
- The laptop and desktop deployment hosts are physically separate computers. Docker volumes and bind-mounted paths are local to each host; never assume weights or files on one are available on the other. Each host needs its own `model_weights` folder and any bind-mounted FaceFusion source file beside that host's Compose file, with required files or weights copied/downloaded separately.
- For face-swap model identifiers and endpoint changes, follow `.github/instructions/face-swap.instructions.md`.

## Company References

- No relevant company-specific guidance for face-swap model naming or the separate laptop/desktop Docker storage setup was found in Confluence. These are project- and user-directed.
