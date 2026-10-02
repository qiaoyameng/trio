from __future__ import annotations

from functools import partial

import pytest
from outcome import Error, Value

import trio
from trio import TaskHandle


class _Sentinel:
    def __repr__(self) -> str:
        return "<sentinel>"


async def test_basic_result_identity() -> None:
    sentinel = _Sentinel()

    async def child() -> _Sentinel:
        await trio.sleep(0)
        return sentinel

    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(child)
        assert isinstance(handle, TaskHandle)
        assert not handle.done
        with pytest.raises(trio.WouldBlock):
            handle.outcome
        result = await handle.wait()
        assert result is sentinel
        assert handle.done
        assert isinstance(handle.outcome, Value)
        assert handle.outcome.value is sentinel
        # Repeated waits return the exact same object.
        assert await handle.wait() is sentinel


async def test_two_waiters_share_result_identity() -> None:
    sentinel = _Sentinel()
    release = trio.Event()
    results: list[object] = []

    async def child() -> _Sentinel:
        await release.wait()
        return sentinel

    async def waiter(handle: TaskHandle[_Sentinel]) -> None:
        results.append(await handle.wait())

    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(child)
        nursery.start_soon(waiter, handle)
        nursery.start_soon(waiter, handle)
        await trio.testing.wait_all_tasks_blocked()
        assert not handle.done
        assert results == []
        release.set()

    assert len(results) == 2
    assert results[0] is sentinel
    assert results[1] is sentinel
    assert results[0] is results[1]


async def test_join_does_not_raise_and_outcome_persists() -> None:
    boom = ValueError("boom")

    async def child() -> None:
        await trio.sleep(0)
        raise boom

    handle_holder: list[TaskHandle[None]] = []

    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with trio.open_nursery() as nursery:
            handle = nursery.start_soon_with_handle(child)
            handle_holder.append(handle)
            await handle.join()
            assert handle.done
            # join() doesn't raise, even though the child failed.
            outcome_result = handle.outcome
            assert isinstance(outcome_result, Error)
            assert outcome_result.error is boom
    # And the nursery still surfaces the same failure on exit.
    (raised,) = excinfo.value.exceptions
    assert raised is boom
    assert handle_holder[0].outcome.error is boom


async def test_wait_raises_same_exception_every_time() -> None:
    boom = KeyError("boom")
    observed: list[BaseException] = []

    async def child() -> None:
        raise boom

    async def observe(handle: TaskHandle[None]) -> None:
        try:
            await handle.wait()
        except BaseException as exc:
            observed.append(exc)

    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with trio.open_nursery() as nursery:
            handle = nursery.start_soon_with_handle(child)
            nursery.start_soon(observe, handle)
            await handle.wait()
    (child_exc,) = excinfo.value.exceptions
    assert child_exc is boom

    # The other waiter saw the exact same exception object, and it remains
    # readable on the handle afterwards.
    assert observed == [boom]
    assert isinstance(handle.outcome, Error)
    assert handle.outcome.error is boom


async def test_cancelled_wait_does_not_cancel_child() -> None:
    release = trio.Event()

    async def child() -> str:
        await release.wait()
        return "done"

    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(child)
        with trio.move_on_after(10) as scope:
            await handle.wait()
        assert scope.cancelled_caught
        # The cancellation hit only our wait; the child is still alive.
        assert not handle.done
        assert handle.task in nursery.child_tasks
        release.set()
        assert await handle.wait() == "done"


async def test_handle_cancel_cancels_only_that_task() -> None:
    victim_started = trio.Event()

    async def victim() -> None:
        victim_started.set()
        await trio.sleep_forever()

    async def survivor() -> None:
        await trio.sleep(0)

    async with trio.open_nursery() as nursery:
        victim_handle = nursery.start_soon_with_handle(victim, name="victim")
        survivor_handle = nursery.start_soon_with_handle(survivor)
        await victim_started.wait()
        await survivor_handle.wait()
        victim_handle.cancel()
        await victim_handle.join()
        # The cancelled task completes with no result and no error leaks.
        assert victim_handle.outcome == Value(None)
        assert survivor_handle.outcome == Value(None)


async def test_handle_cancel_before_start_is_safe_and_idempotent() -> None:
    async def child() -> int:
        return 42

    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(child)
        handle.cancel()
        handle.cancel()
        await handle.join()
        assert handle.done


async def test_failed_child_still_raises_from_nursery_strict() -> None:
    boom = RuntimeError("child failure")

    async def child() -> None:
        await trio.sleep(0)
        raise boom

    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with trio.open_nursery() as nursery:
            handle = nursery.start_soon_with_handle(child)
            # Reading the failure early must not swallow it.
            with pytest.raises(RuntimeError) as early:
                await handle.wait()
            assert early.value is boom
    (raised,) = excinfo.value.exceptions
    assert raised is boom


async def test_nursery_wide_cancellation_behaves_normally() -> None:
    async def forever() -> None:
        await trio.sleep_forever()

    with trio.move_on_after(1):
        async with trio.open_nursery() as nursery:
            handle = nursery.start_soon_with_handle(forever)
            await handle.join()
    assert handle.done


async def test_start_after_nursery_closed_raises() -> None:
    async def child() -> None:
        return None

    nursery_manager = trio.open_nursery()
    nursery = await nursery_manager.__aenter__()
    try:
        nursery.start_soon_with_handle(child)
    finally:
        await nursery_manager.__aexit__(None, None, None)
    with pytest.raises(RuntimeError, match="closed"):
        nursery.start_soon_with_handle(child)


async def test_startup_failures_raise_synchronously() -> None:
    async def coro_fn() -> None:
        pass

    coro_object = coro_fn()
    try:
        async with trio.open_nursery() as nursery:
            # Passing a coroutine *object* instead of the function.
            with pytest.raises(TypeError, match="async function"):
                nursery.start_soon_with_handle(coro_object)
            # A plain sync callable is rejected immediately too.
            with pytest.raises(TypeError):
                nursery.start_soon_with_handle(lambda: 1)
            # Neither bad call scheduled anything.
            assert nursery.child_tasks == frozenset()
    finally:
        coro_object.close()


async def test_kwargs_via_partial() -> None:
    async def child(*, value: object) -> object:
        await trio.sleep(0)
        return value

    sentinel = _Sentinel()
    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(partial(child, value=sentinel))
        assert await handle.wait() is sentinel


async def test_named_handle_uses_user_function_name() -> None:
    async def child() -> int:
        return 7

    async with trio.open_nursery() as nursery:
        handle = nursery.start_soon_with_handle(child, name="named-child")
        assert "named-child" in repr(handle.task)
        assert await handle.wait() == 7


async def test_cancel_scope_attribute_is_per_task() -> None:
    started = trio.Event()
    h2_gate = trio.Event()

    async def child() -> None:
        started.set()
        await trio.sleep_forever()

    async def child2() -> None:
        await h2_gate.wait()

    with trio.CancelScope(shield=True) as parent_shield:
        async with trio.open_nursery() as nursery:
            h1 = nursery.start_soon_with_handle(child, name="c1")
            h2 = nursery.start_soon_with_handle(child2, name="c2")
            assert h1.cancel_scope is not h2.cancel_scope
            assert h1.cancel_scope is not nursery.cancel_scope
            await started.wait()
            h1.cancel()
            await h1.join()
            # c1's cancellation must not have touched the nursery or c2.
            assert not nursery.cancel_scope.cancel_called
            assert not h2.done
            h2_gate.set()
            assert await h2.wait() is None
    assert parent_shield is not None
