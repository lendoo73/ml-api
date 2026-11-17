import inspect
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast


Function = TypeVar("Function", bound=Callable[..., Any])


def execution_time(message: str = "Execution time:") -> Callable[[Function], Function]:
    def decorator(function: Function) -> Function:
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await function(*args, **kwargs)
                finally:
                    print(f"{message} {time.perf_counter() - start:.6f}s")

            return cast(Function, async_wrapper)

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                print(f"{message} {time.perf_counter() - start:.6f}s")

        return cast(Function, wrapper)

    return decorator
