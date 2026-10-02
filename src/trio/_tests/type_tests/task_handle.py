"""Check the types of Nursery.start_soon_with_handle / TaskHandle."""

from outcome import Outcome

import trio
from trio import Nursery, TaskHandle
from typing_extensions import assert_type


async def returns_int() -> int:
    return 42


async def returns_none() -> None:
    return None


async def check(nursery: Nursery) -> None:
    int_handle = nursery.start_soon_with_handle(returns_int)
    none_handle = nursery.start_soon_with_handle(returns_none)

    assert_type(int_handle, TaskHandle[int])
    assert_type(none_handle, TaskHandle[None])

    assert_type(int_handle.done, bool)
    assert_type(int_handle.outcome, Outcome[int])
    assert_type(int_handle.cancel_scope, trio.CancelScope)

    assert_type(await int_handle.wait(), int)
    assert_type(await none_handle.wait(), None)
    await int_handle.join()

    int_handle.cancel()
    int_handle.cancel("because")
