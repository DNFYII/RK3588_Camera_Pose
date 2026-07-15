import os

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
