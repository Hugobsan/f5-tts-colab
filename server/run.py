# Entry point: `python run.py`. Kept import-free at module level because spawned workers
# re-run the main script (as __mp_main__), possibly under an engine venv without fastapi.
if __name__ == "__main__":
    import f5_server

    f5_server.main()
