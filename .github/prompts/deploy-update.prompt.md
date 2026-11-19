---
mode: agent
description: Test, publish, and deploy the current ML-API update to the laptop's Cloudflare-backed Docker deployment.
---

# Deploy this update

Use this prompt when the user explicitly asks to deploy the current update. Carry the update through source-control publication, Docker Hub publishing, Compose synchronization, and redeployment on this laptop. Do not stop after only building or pushing an image.

## 1. Inspect and validate

- Inspect the worktree and current branch first. Preserve unrelated user changes and never include generated media, model weights, secrets, or unrelated files in a commit.
- Review the actual Git remotes, current branch, project version/tag, Dockerfile, `.dockerignore`, and every relevant Compose file. This repository's configured Git remote is named `origin`; use its actual URL rather than assuming a hosting provider.
- Run the narrowest relevant tests and checks for the update. Fix failures caused by the update; report unrelated existing failures accurately.
- Ensure model weights remain outside the Docker build context and are persisted in the existing named `model_weights` volume mounted at `/app/weights`.
- If frontend source changes and the sibling React source is available, rebuild the served `build` assets using that project's existing build workflow before creating the API image. Do not make unrelated frontend changes.

## 2. Commit and push source changes

- Stage only files belonging to this update; inspect the staged diff for secrets, weight files, generated media, and unrelated changes.
- Commit the update using the repository's commit conventions, then push the current branch to the configured `origin`.
- Do not force-push, amend existing commits, or discard user changes.

## 3. Build and publish the Docker image

- Determine the API image repository and release tag from the current laptop Cloudflare Compose file and project version. The current deployment uses `lendoo73/ml-rest-api`; do not assume the README's older image name if deployment configuration differs.
- Do not overwrite an existing release tag. Do not create or move `latest` unless the user explicitly asks.
- Build the API image from the repository root with the repository Dockerfile and `.dockerignore`. Also build and push `lendoo73/ml-rest-api-video-worker` using `face_swap_video_worker/Dockerfile`; keep its release tag synchronized with the API release tag unless an existing tag would be overwritten. Push both selected tags to Docker Hub. Do not load the huge API image into the local Docker image store if a direct Buildx push is available and appropriate.
- If an image-build or registry push fails, resolve the concrete failure and retry; do not claim publication until the push succeeds.

## 4. Synchronize deployment Compose files

- Check whether the image tag or runtime configuration changed. Update only Compose files that should track this deployment:
  - Repository deployment: `deploy/docker-compose.yaml`.
  - Laptop Cloudflare deployment: `C:\Docker\Images\dockerhub\lendoo73\ml-rest-api\docker-compose.yaml`.
  - Desktop deployment copy, when present: `C:\Docker\Images\dockerhub\lendoo73\ml-rest-api\desktop\docker-compose.yaml`.
- Ensure each deployment Compose has the `face-swap-video-worker` service using the matching published worker image, GPU access, and the same named `/app/weights` volume as the API. Configure the API's `FACE_SWAP_VIDEO_WORKER_URL` to the internal worker service and make API startup wait for the worker health check.
- Keep GPU access (`gpus: all`) on both inference containers, set `FACE_SWAP_VIDEO_WORKER_URL` on the API to the internal worker service, retain the Cloudflare tunnel, and mount the same persistent named weights volume at `/app/weights` in both containers. Keep the desktop port/configuration differences intact.
- Validate the changed Compose YAML. Do not require local `.env` files to be committed or fabricate secrets.

## 5. Redeploy this laptop

- From `C:\Docker\Images\dockerhub\lendoo73\ml-rest-api`, pull the selected image and recreate the Cloudflare-backed Compose services with `docker compose pull` and `docker compose up -d --force-recreate`.
- Never use `docker compose down -v`, remove the named weights volume, or clear model data during routine deployment.
- Confirm the API, FaceFusion video worker, and Cloudflare tunnel containers are running; the API responds at `http://localhost:8010/api/docs` and through the configured public API domain; both inference containers have GPU access; and `/app/weights` is still mounted from the existing named volume with its contents intact.

## 6. Remove only deployment-created temporary image tags

- After the new deployment passes verification, inspect local image tags and container references.
- Remove temporary tags created for this deployment (for example `:test`) only after the release image is successfully published and no container references those tags. Do not remove the deployed release tag or images needed by any running or stopped container.
- Do not run broad image, container, build-cache, or volume prune commands as part of this prompt. If cleanup would affect unrelated Docker workloads or requires removing an image still referenced by a container, stop and ask the user.

## Completion report

Report the source commit and push result, Docker Hub repository/tag and push result, Compose files changed, laptop and Cloudflare verification results, and any temporary tags removed. Clearly state anything blocked or not completed. Never report success for a step that did not complete.
