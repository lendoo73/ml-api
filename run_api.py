import os


CUDA_TOOLKIT_PATHS = {
    os.path.normcase(
        os.path.normpath(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin")
    ),
    os.path.normcase(
        os.path.normpath(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
    ),
}


def _remove_cuda_toolkit_paths() -> int:
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    filtered_entries = [
        entry
        for entry in path_entries
        if os.path.normcase(os.path.normpath(entry)) not in CUDA_TOOLKIT_PATHS
    ]
    os.environ["PATH"] = os.pathsep.join(filtered_entries)
    return len(path_entries) - len(filtered_entries)


def main() -> None:
    removed_count = _remove_cuda_toolkit_paths()
    if removed_count:
        print(f"Removed {removed_count} CUDA Toolkit PATH entr{'y' if removed_count == 1 else 'ies'}.")

    import uvicorn

    uvicorn.run("main:app", reload=True)


if __name__ == "__main__":
    main()
