from __future__ import annotations

import re
import weakref
from collections.abc import Callable
from typing import TypeAlias

import pytest

from .. import _core
from .._core._parking_lot import GLOBAL_PARKING_LOT_BREAKER
from .._sync import *
from .._timeouts import sleep_forever
from ..testing import assert_checkpoints, wait_all_tasks_blocked


async def test_Event() -> None:
    e = Event()
    assert not e.is_set()
    assert e.statistics().tasks_waiting == 0

    with pytest.raises(NotImplementedError):
        e.__bool__()

    e.set()
    assert e.is_set()
    with assert_checkpoints():
        await e.wait()

    e = Event()

    record = []

    async def child() -> None:
        record.append("sleeping")
        await e.wait()
        record.append("woken")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child)
        nursery.start_soon(child)
        await wait_all_tasks_blocked()
        assert record == ["sleeping", "sleeping"]
        assert e.statistics().tasks_waiting == 2
        e.set()
        await wait_all_tasks_blocked()
        assert record == ["sleeping", "sleeping", "woken", "woken"]


async def test_CapacityLimiter() -> None:
    assert CapacityLimiter(0).total_tokens == 0
    with pytest.raises(TypeError):
        CapacityLimiter(1.0)
    with pytest.raises(ValueError, match=r"^total_tokens must be >= 0$"):
        CapacityLimiter(-1)
    c = CapacityLimiter(2)
    repr(c)  # smoke test
    assert c.total_tokens == 2
    assert c.borrowed_tokens == 0
    assert c.available_tokens == 2
    with pytest.raises(RuntimeError):
        c.release()
    assert c.borrowed_tokens == 0
    c.acquire_nowait()
    assert c.borrowed_tokens == 1
    assert c.available_tokens == 1

    stats = c.statistics()
    assert stats.borrowed_tokens == 1
    assert stats.total_tokens == 2
    assert stats.borrowers == [_core.current_task()]
    assert stats.tasks_waiting == 0

    # Can't re-acquire when we already have it
    with pytest.raises(RuntimeError):
        c.acquire_nowait()
    assert c.borrowed_tokens == 1
    with pytest.raises(RuntimeError):
        await c.acquire()
    assert c.borrowed_tokens == 1

    # We can acquire on behalf of someone else though
    with assert_checkpoints():
        await c.acquire_on_behalf_of("someone")

    # But then we've run out of capacity
    assert c.borrowed_tokens == 2
    with pytest.raises(_core.WouldBlock):
        c.acquire_on_behalf_of_nowait("third party")

    assert set(c.statistics().borrowers) == {_core.current_task(), "someone"}

    # Until we release one
    c.release_on_behalf_of(_core.current_task())
    assert c.statistics().borrowers == ["someone"]

    c.release_on_behalf_of("someone")
    assert c.borrowed_tokens == 0
    with assert_checkpoints():
        async with c:
            assert c.borrowed_tokens == 1

    async with _core.open_nursery() as nursery:
        await c.acquire_on_behalf_of("value 1")
        await c.acquire_on_behalf_of("value 2")
        nursery.start_soon(c.acquire_on_behalf_of, "value 3")
        await wait_all_tasks_blocked()
        assert c.borrowed_tokens == 2
        assert c.statistics().tasks_waiting == 1
        c.release_on_behalf_of("value 2")
        # Fairness:
        assert c.borrowed_tokens == 2
        with pytest.raises(_core.WouldBlock):
            c.acquire_nowait()

    c.release_on_behalf_of("value 3")
    c.release_on_behalf_of("value 1")


async def test_CapacityLimiter_inf() -> None:
    from math import inf

    c = CapacityLimiter(inf)
    repr(c)  # smoke test
    assert c.total_tokens == inf
    assert c.borrowed_tokens == 0
    assert c.available_tokens == inf
    with pytest.raises(RuntimeError):
        c.release()
    assert c.borrowed_tokens == 0
    c.acquire_nowait()
    assert c.borrowed_tokens == 1
    assert c.available_tokens == inf


async def test_CapacityLimiter_change_total_tokens() -> None:
    c = CapacityLimiter(2)

    with pytest.raises(TypeError):
        c.total_tokens = 1.0

    with pytest.raises(ValueError, match=r"^total_tokens must be >= 0$"):
        c.total_tokens = -1

    with pytest.raises(ValueError, match=r"^total_tokens must be >= 0$"):
        c.total_tokens = -10

    assert c.total_tokens == 2

    async with _core.open_nursery() as nursery:
        for i in range(5):
            nursery.start_soon(c.acquire_on_behalf_of, i)
            await wait_all_tasks_blocked()
        assert set(c.statistics().borrowers) == {0, 1}
        assert c.statistics().tasks_waiting == 3
        c.total_tokens += 2
        assert set(c.statistics().borrowers) == {0, 1, 2, 3}
        assert c.statistics().tasks_waiting == 1
        c.total_tokens -= 3
        assert c.borrowed_tokens == 4
        assert c.total_tokens == 1
        c.release_on_behalf_of(0)
        c.release_on_behalf_of(1)
        c.release_on_behalf_of(2)
        assert set(c.statistics().borrowers) == {3}
        assert c.statistics().tasks_waiting == 1
        c.release_on_behalf_of(3)
        assert set(c.statistics().borrowers) == {4}
        assert c.statistics().tasks_waiting == 0


# regression test for issue #548
async def test_CapacityLimiter_memleak_548() -> None:
    limiter = CapacityLimiter(total_tokens=1)
    await limiter.acquire()

    async with _core.open_nursery() as n:
        n.start_soon(limiter.acquire)
        await wait_all_tasks_blocked()  # give it a chance to run the task
        n.cancel_scope.cancel()

    # if this is 1, the acquire call (despite being killed) is still there in the task, and will
    # leak memory all the while the limiter is active
    assert len(limiter._pending_borrowers) == 0


async def test_CapacityLimiter_zero_limit_tokens() -> None:
    c = CapacityLimiter(5)

    assert c.total_tokens == 5

    async with _core.open_nursery() as nursery:
        c.total_tokens = 0

        for i in range(5):
            nursery.start_soon(c.acquire_on_behalf_of, i)
            await wait_all_tasks_blocked()

        assert set(c.statistics().borrowers) == set()
        assert c.statistics().tasks_waiting == 5

        c.total_tokens = 5

        assert set(c.statistics().borrowers) == {0, 1, 2, 3, 4}

        nursery.start_soon(c.acquire_on_behalf_of, 5)
        await wait_all_tasks_blocked()

        assert c.statistics().tasks_waiting == 1

        for i in range(5):
            c.release_on_behalf_of(i)

        assert c.statistics().tasks_waiting == 0
        c.release_on_behalf_of(5)

        # making sure that zero limit capacity limiter doesn't let any tasks through

        c.total_tokens = 0

        with pytest.raises(_core.WouldBlock):
            c.acquire_nowait()

        nursery.start_soon(c.acquire_on_behalf_of, 6)
        await wait_all_tasks_blocked()

        assert c.statistics().tasks_waiting == 1
        assert c.statistics().borrowers == []

        c.total_tokens = 1
        assert c.statistics().tasks_waiting == 0
        assert c.statistics().borrowers == [6]
        c.release_on_behalf_of(6)

        await c.acquire_on_behalf_of(0)  # total_tokens is 1

        nursery.start_soon(c.acquire_on_behalf_of, 1)
        await wait_all_tasks_blocked()
        c.total_tokens = 0

        assert c.statistics().borrowers == [0]

        c.release_on_behalf_of(0)
        await wait_all_tasks_blocked()
        assert c.statistics().borrowers == []
        assert c.statistics().tasks_waiting == 1

        c.total_tokens = 1
        await wait_all_tasks_blocked()
        assert c.statistics().borrowers == [1]
        assert c.statistics().tasks_waiting == 0

        c.release_on_behalf_of(1)

        c.total_tokens = 0

        nursery.cancel_scope.cancel()

    assert c.total_tokens == 0
    assert c.statistics().borrowers == []
    assert c._pending_borrowers == {}


async def test_Semaphore() -> None:
    with pytest.raises(TypeError):
        Semaphore(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^initial value must be >= 0$"):
        Semaphore(-1)
    s = Semaphore(1)
    repr(s)  # smoke test
    assert s.value == 1
    assert s.max_value is None
    s.release()
    assert s.value == 2
    assert s.statistics().tasks_waiting == 0
    s.acquire_nowait()
    assert s.value == 1
    with assert_checkpoints():
        await s.acquire()
    assert s.value == 0
    with pytest.raises(_core.WouldBlock):
        s.acquire_nowait()

    s.release()
    assert s.value == 1
    with assert_checkpoints():
        async with s:
            assert s.value == 0
    assert s.value == 1
    s.acquire_nowait()

    record = []

    async def do_acquire(s: Semaphore) -> None:
        record.append("started")
        await s.acquire()
        record.append("finished")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(do_acquire, s)
        await wait_all_tasks_blocked()
        assert record == ["started"]
        assert s.value == 0
        s.release()
        # Fairness:
        assert s.value == 0
        with pytest.raises(_core.WouldBlock):
            s.acquire_nowait()
    assert record == ["started", "finished"]


def test_Semaphore_bounded() -> None:
    with pytest.raises(TypeError):
        Semaphore(1, max_value=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^max_values must be >= initial_value$"):
        Semaphore(2, max_value=1)
    bs = Semaphore(1, max_value=1)
    assert bs.max_value == 1
    repr(bs)  # smoke test
    with pytest.raises(ValueError, match=r"^semaphore released too many times$"):
        bs.release()
    assert bs.value == 1
    bs.acquire_nowait()
    assert bs.value == 0
    bs.release()
    assert bs.value == 1


@pytest.mark.parametrize("lockcls", [Lock, StrictFIFOLock], ids=lambda fn: fn.__name__)
async def test_Lock_and_StrictFIFOLock(
    lockcls: type[Lock | StrictFIFOLock],
) -> None:
    l = lockcls()  # noqa
    assert not l.locked()

    # make sure locks can be weakref'ed (gh-331)
    r = weakref.ref(l)
    assert r() is l

    repr(l)  # smoke test
    # make sure repr uses the right name for subclasses
    assert lockcls.__name__ in repr(l)
    with assert_checkpoints():
        async with l:
            assert l.locked()
            repr(l)  # smoke test (repr branches on locked/unlocked)
    assert not l.locked()
    l.acquire_nowait()
    assert l.locked()
    l.release()
    assert not l.locked()
    with assert_checkpoints():
        await l.acquire()
    assert l.locked()
    l.release()
    assert not l.locked()

    l.acquire_nowait()
    with pytest.raises(RuntimeError):
        # Error out if we already own the lock
        l.acquire_nowait()
    l.release()
    with pytest.raises(RuntimeError):
        # Error out if we don't own the lock
        l.release()

    holder_task = None

    async def holder() -> None:
        nonlocal holder_task
        holder_task = _core.current_task()
        async with l:
            await sleep_forever()

    async with _core.open_nursery() as nursery:
        assert not l.locked()
        nursery.start_soon(holder)
        await wait_all_tasks_blocked()
        assert l.locked()
        # WouldBlock if someone else holds the lock
        with pytest.raises(_core.WouldBlock):
            l.acquire_nowait()
        # Can't release a lock someone else holds
        with pytest.raises(RuntimeError):
            l.release()

        statistics = l.statistics()
        print(statistics)
        assert statistics.locked
        assert statistics.owner is holder_task
        assert statistics.tasks_waiting == 0

        nursery.start_soon(holder)
        await wait_all_tasks_blocked()
        statistics = l.statistics()
        print(statistics)
        assert statistics.tasks_waiting == 1

        nursery.cancel_scope.cancel()

    statistics = l.statistics()
    assert not statistics.locked
    assert statistics.owner is None
    assert statistics.tasks_waiting == 0


async def test_Condition() -> None:
    with pytest.raises(TypeError):
        Condition(Semaphore(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Condition(StrictFIFOLock)  # type: ignore[arg-type]
    l = Lock()  # noqa
    c = Condition(l)
    assert not l.locked()
    assert not c.locked()
    with assert_checkpoints():
        await c.acquire()
    assert l.locked()
    assert c.locked()

    c = Condition()
    assert not c.locked()
    c.acquire_nowait()
    assert c.locked()
    with pytest.raises(RuntimeError):
        c.acquire_nowait()
    c.release()

    with pytest.raises(RuntimeError):
        # Can't wait without holding the lock
        await c.wait()
    with pytest.raises(RuntimeError):
        # Can't notify without holding the lock
        c.notify()
    with pytest.raises(RuntimeError):
        # Can't notify without holding the lock
        c.notify_all()

    finished_waiters = set()

    async def waiter(i: int) -> None:
        async with c:
            await c.wait()
        finished_waiters.add(i)

    async with _core.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(waiter, i)
            await wait_all_tasks_blocked()
        async with c:
            c.notify()
        assert c.locked()
        await wait_all_tasks_blocked()
        assert finished_waiters == {0}
        async with c:
            c.notify_all()
        await wait_all_tasks_blocked()
        assert finished_waiters == {0, 1, 2}

    finished_waiters = set()
    async with _core.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(waiter, i)
            await wait_all_tasks_blocked()
        async with c:
            c.notify(2)
            statistics = c.statistics()
            print(statistics)
            assert statistics.tasks_waiting == 1
            assert statistics.lock_statistics.tasks_waiting == 2
        # exiting the context manager hands off the lock to the first task
        assert c.statistics().lock_statistics.tasks_waiting == 1

        await wait_all_tasks_blocked()
        assert finished_waiters == {0, 1}

        async with c:
            c.notify_all()

    # After being cancelled still hold the lock (!)
    # (Note that c.__aexit__ checks that we hold the lock as well)
    with _core.CancelScope() as scope:
        async with c:
            scope.cancel()
            try:
                await c.wait()
            finally:
                assert c.locked()


from .._channel import open_memory_channel
from .._sync import AsyncContextManagerMixin

# Three ways of implementing a Lock in terms of a channel. Used to let us put
# the channel through the generic lock tests.


class ChannelLock1(AsyncContextManagerMixin):
    def __init__(self, capacity: int) -> None:
        self.s, self.r = open_memory_channel[None](capacity)
        for _ in range(capacity - 1):
            self.s.send_nowait(None)

    def acquire_nowait(self) -> None:
        self.s.send_nowait(None)

    async def acquire(self) -> None:
        await self.s.send(None)

    def release(self) -> None:
        self.r.receive_nowait()


class ChannelLock2(AsyncContextManagerMixin):
    def __init__(self) -> None:
        self.s, self.r = open_memory_channel[None](10)
        self.s.send_nowait(None)

    def acquire_nowait(self) -> None:
        self.r.receive_nowait()

    async def acquire(self) -> None:
        await self.r.receive()

    def release(self) -> None:
        self.s.send_nowait(None)


class ChannelLock3(AsyncContextManagerMixin):
    def __init__(self) -> None:
        self.s, self.r = open_memory_channel[None](0)
        # self.acquired is true when one task acquires the lock and
        # only becomes false when it's released and no tasks are
        # waiting to acquire.
        self.acquired = False

    def acquire_nowait(self) -> None:
        assert not self.acquired
        self.acquired = True

    async def acquire(self) -> None:
        if self.acquired:
            await self.s.send(None)
        else:
            self.acquired = True
            await _core.checkpoint()

    def release(self) -> None:
        try:
            self.r.receive_nowait()
        except _core.WouldBlock:
            assert self.acquired
            self.acquired = False


lock_factories = [
    lambda: CapacityLimiter(1),
    lambda: Semaphore(1),
    Lock,
    StrictFIFOLock,
    lambda: ChannelLock1(10),
    lambda: ChannelLock1(1),
    ChannelLock2,
    ChannelLock3,
]
lock_factory_names = [
    "CapacityLimiter(1)",
    "Semaphore(1)",
    "Lock",
    "StrictFIFOLock",
    "ChannelLock1(10)",
    "ChannelLock1(1)",
    "ChannelLock2",
    "ChannelLock3",
]

generic_lock_test = pytest.mark.parametrize(
    "lock_factory",
    lock_factories,
    ids=lock_factory_names,
)

LockLike: TypeAlias = (
    CapacityLimiter
    | Semaphore
    | Lock
    | StrictFIFOLock
    | ChannelLock1
    | ChannelLock2
    | ChannelLock3
)
LockFactory: TypeAlias = Callable[[], LockLike]


# Spawn a bunch of workers that take a lock and then yield; make sure that
# only one worker is ever in the critical section at a time.
@generic_lock_test
async def test_generic_lock_exclusion(lock_factory: LockFactory) -> None:
    LOOPS = 10
    WORKERS = 5
    in_critical_section = False
    acquires = 0

    async def worker(lock_like: LockLike) -> None:
        nonlocal in_critical_section, acquires
        for _ in range(LOOPS):
            async with lock_like:
                acquires += 1
                assert not in_critical_section
                in_critical_section = True
                await _core.checkpoint()
                await _core.checkpoint()
                assert in_critical_section
                in_critical_section = False

    async with _core.open_nursery() as nursery:
        lock_like = lock_factory()
        for _ in range(WORKERS):
            nursery.start_soon(worker, lock_like)
    assert not in_critical_section
    assert acquires == LOOPS * WORKERS


# Several workers queue on the same lock; make sure they each get it, in
# order.
@generic_lock_test
async def test_generic_lock_fifo_fairness(lock_factory: LockFactory) -> None:
    initial_order = []
    record = []
    LOOPS = 5

    async def loopy(name: int, lock_like: LockLike) -> None:
        # Record the order each task was initially scheduled in
        initial_order.append(name)
        for _ in range(LOOPS):
            async with lock_like:
                record.append(name)

    lock_like = lock_factory()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(loopy, 1, lock_like)
        nursery.start_soon(loopy, 2, lock_like)
        nursery.start_soon(loopy, 3, lock_like)
    # The first three could be in any order due to scheduling randomness,
    # but after that they should repeat in the same order
    for i in range(LOOPS):
        assert record[3 * i : 3 * (i + 1)] == initial_order


@generic_lock_test
async def test_generic_lock_acquire_nowait_blocks_acquire(
    lock_factory: LockFactory,
) -> None:
    lock_like = lock_factory()

    record = []

    async def lock_taker() -> None:
        record.append("started")
        async with lock_like:
            pass
        record.append("finished")

    async with _core.open_nursery() as nursery:
        lock_like.acquire_nowait()
        nursery.start_soon(lock_taker)
        await wait_all_tasks_blocked()
        assert record == ["started"]
        lock_like.release()


async def test_lock_acquire_unowned_lock() -> None:
    """Test that trying to acquire a lock whose owner has exited raises an error.
    see https://github.com/python-trio/trio/issues/3035
    """
    assert not GLOBAL_PARKING_LOT_BREAKER
    lock = trio.Lock()
    async with trio.open_nursery() as nursery:
        nursery.start_soon(lock.acquire)
    owner_str = re.escape(str(lock._lot.broken_by[0]))
    with pytest.raises(
        trio.BrokenResourceError,
        match=f"^Owner of this lock exited without releasing: {owner_str}$",
    ):
        await lock.acquire()
    assert not GLOBAL_PARKING_LOT_BREAKER


async def test_lock_multiple_acquire() -> None:
    """Test for error if awaiting on a lock whose owner exits without releasing.
    see https://github.com/python-trio/trio/issues/3035"""
    assert not GLOBAL_PARKING_LOT_BREAKER
    lock = trio.Lock()
    with pytest.RaisesGroup(
        pytest.RaisesExc(
            trio.BrokenResourceError,
            match="^Owner of this lock exited without releasing: ",
        ),
    ):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(lock.acquire)
            nursery.start_soon(lock.acquire)
    assert not GLOBAL_PARKING_LOT_BREAKER


async def test_lock_handover() -> None:
    assert not GLOBAL_PARKING_LOT_BREAKER
    child_task: Task | None = None
    lock = trio.Lock()

    # this task acquires the lock
    lock.acquire_nowait()
    assert {
        _core.current_task(): [
            lock._lot,
        ],
    } == GLOBAL_PARKING_LOT_BREAKER

    async with trio.open_nursery() as nursery:
        nursery.start_soon(lock.acquire)
        await wait_all_tasks_blocked()

        # hand over the lock to the child task
        lock.release()

        # check values, and get the identifier out of the dict for later check
        assert len(GLOBAL_PARKING_LOT_BREAKER) == 1
        child_task = next(iter(GLOBAL_PARKING_LOT_BREAKER))
        assert GLOBAL_PARKING_LOT_BREAKER[child_task] == [lock._lot]

    assert lock._lot.broken_by == [child_task]
    assert not GLOBAL_PARKING_LOT_BREAKER




async def test_Barrier_invalid_parties() -> None:
    with pytest.raises(TypeError):
        Barrier(3.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        # bool is a subtype of int but is not a valid participant count
        Barrier(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^parties must be >= 1$"):
        Barrier(0)
    with pytest.raises(ValueError, match=r"^parties must be >= 1$"):
        Barrier(-2)


async def test_Barrier_single_party_is_coordinator_and_checkpoint() -> None:
    b = Barrier(1)
    assert b.statistics() == BarrierStatistics(
        parties=1,
        tasks_waiting=0,
        generation=0,
        broken=False,
    )
    with assert_checkpoints():
        result = await b.wait()
    assert result.coordinator
    assert result.party == 0
    assert result.generation == 0
    result2 = await b.wait()
    assert result2.coordinator
    assert result2.generation == 1
    assert b.generation == 2
    assert b.n_waiting == 0


async def test_Barrier_wait_cancelled_before_entering_does_not_break() -> None:
    b = Barrier(2)
    with trio.move_on_at(float("-inf")):
        await b.wait()
    assert not b.broken
    assert b.n_waiting == 0
    assert b.generation == 0
    # The barrier still completes a normal generation afterwards.
    box: list[BarrierWaitResult] = []

    async def one() -> None:
        box.append(await b.wait())

    async with _core.open_nursery() as nursery:
        nursery.start_soon(one)
        nursery.start_soon(one)
    assert sum(r.coordinator for r in box) == 1


class _CancelLastArriverInstrument:
    """Cancel ``scope`` at the exact step where the last participant has
    registered but has not yet crossed the barrier's release checkpoint.

    The last task is in the generation's waiting set but is not parked in
    the generation's parking lot, while the other ``parties - 1`` tasks are.
    That unique state makes the injection point deterministic.
    """

    def __init__(self, barrier: Barrier, scope: trio.CancelScope) -> None:
        self._barrier = barrier
        self._scope = scope
        self.armed = False
        self.fired = False

    def _scope_is_active_for(self, task: _core.Task) -> bool:
        status = task._cancel_status
        while status is not None:
            if status._scope is self._scope:
                return True
            status = getattr(status, "_parent", None)
        return False

    def before_task_step(self, task: _core.Task) -> None:
        if not self.armed or self.fired:
            return
        gen = self._barrier._gen  # type: ignore[attr-defined]
        # The last arriver is the task whose step makes the waiting set
        # full while the other ``parties - 1`` tasks are still parked. We
        # additionally confirm the targeted cancel scope is active for the
        # stepping task, which rules out the controlling task.
        if (
            gen is self._barrier._gen
            and len(gen.waiting) == self._barrier.parties
            and len(gen.lot._parked) == self._barrier.parties - 1  # type: ignore[attr-defined]
            and task in gen.waiting
            and task not in gen.lot._parked  # type: ignore[attr-defined]
            and self._scope_is_active_for(task)
        ):
            self.fired = True
            self.armed = False
            self._scope.cancel()


async def test_Barrier_controlled_four_rounds_with_cancel_and_reset() -> None:
    parties = 3
    b = Barrier(parties)

    # admit[round][index] gates admission of one participant into one round,
    # so the controller can admit the first parties-1 tasks, let them park,
    # and only then admit the final participant.
    admit: list[list[trio.Event]] = [
        [trio.Event() for _ in range(parties)] for _ in range(4)
    ]
    # The cancel scope for round 3 is created before the participant waits
    # on its admission gate, so it exists for the instrument to target while
    # the final participant is still outside the barrier.
    round3_scopes = {i: trio.CancelScope() for i in range(parties)}
    outcomes: dict[int, list[str]] = {i: [] for i in range(parties)}

    async def participant(index: int) -> None:
        for round_index in range(4):
            await admit[round_index][index].wait()
            if round_index == 2:
                scope = round3_scopes[index]
            else:
                scope = trio.CancelScope()
            result: BarrierWaitResult | None = None
            broken = False
            with scope:
                try:
                    result = await b.wait()
                except BrokenBarrierError:
                    broken = True
            # A CancelScope swallows its own Cancelled, so that case is read
            # from the scope instead of caught (this also avoids reusing a
            # previous round's ``result`` across loop iterations).
            if scope.cancel_called:
                outcomes[index].append(f"cancelled-r{round_index}")
            elif broken:
                outcomes[index].append(f"broken-r{round_index}")
            else:
                assert result is not None
                outcomes[index].append(
                    f"{'coord' if result.coordinator else 'waiter'}-g{result.generation}",
                )

    async with _core.open_nursery() as nursery:
        for i in range(parties):
            nursery.start_soon(participant, i)

        # ---- Rounds 1 and 2: two clean, fully controlled generations. ----
        for round_index in range(2):
            for i in range(parties):
                admit[round_index][i].set()
            await wait_all_tasks_blocked()
            assert b.n_waiting == 0
            assert not b.broken
            assert b.generation == round_index + 1

        # ---- Round 3 --------------------------------------------------
        gen_before = b.generation

        # Admit the first parties-1 participants and wait until they are
        # parked: waiting set non-full, nobody released yet.
        for i in range(parties - 1):
            admit[2][i].set()
        while b.n_waiting < parties - 1:
            await _core.checkpoint()
        await wait_all_tasks_blocked()
        assert b.n_waiting == parties - 1
        assert not b.broken
        assert b.generation == gen_before

        # Arm before admitting the last participant. The instrument fires at
        # the last arriver's release checkpoint and cancels that generation.
        instrument = _CancelLastArriverInstrument(b, round3_scopes[parties - 1])
        instrument.armed = True
        trio.lowlevel.add_instrument(instrument)
        try:
            admit[2][parties - 1].set()
            await wait_all_tasks_blocked()
        finally:
            trio.lowlevel.remove_instrument(instrument)
        assert instrument.fired

        assert b.broken
        assert b.generation == gen_before
        await wait_all_tasks_blocked()

        # ---- Reset clears the broken generation and starts a new one. ----
        b.reset()
        assert not b.broken
        assert b.n_waiting == 0
        assert b.generation == gen_before + 1

        # ---- Round 4: clean release on the fresh generation. ----
        for i in range(parties):
            admit[3][i].set()
        await wait_all_tasks_blocked()
        assert b.n_waiting == 0
        assert not b.broken
        assert b.generation == gen_before + 2

    # Each participant recorded exactly one outcome per round.
    for i in range(parties):
        assert len(outcomes[i]) == 4, (i, outcomes[i])

    # Successful rounds 1, 2 and 4 produced generations 0, 1 and 3 with one
    # coordinator each.
    coord_generations = sorted(
        int(tag.split("g")[1])
        for i in range(parties)
        for tag in outcomes[i]
        if tag.startswith("coord")
    )
    assert coord_generations == [0, 1, 3]

    # Round 3: exactly one cancellation (the last arriver) and parties-1
    # BrokenBarrierError wakeups -- no success leaked into that round.
    cancelled = [i for i in range(parties) if "cancelled-r2" in outcomes[i]]
    broken = [i for i in range(parties) if "broken-r2" in outcomes[i]]
    assert cancelled == [parties - 1]
    assert sorted(broken) == list(range(parties - 1))

    # Round 4: everyone released in generation 3, exactly one coordinator.
    round4 = [outcomes[i][3] for i in range(parties)]
    assert round4.count("coord-g3") == 1
    assert round4.count("waiter-g3") == parties - 1


async def test_Barrier_abort_wakes_current_generation() -> None:
    parties = 3
    b = Barrier(parties)
    woken: list[int] = []

    async def participant(i: int) -> None:
        with pytest.raises(BrokenBarrierError):
            await b.wait()
        woken.append(i)

    async with _core.open_nursery() as nursery:
        for i in range(parties - 1):
            nursery.start_soon(participant, i)
        await wait_all_tasks_blocked()
        assert b.n_waiting == parties - 1
        assert not b.broken
        b.abort()
        await wait_all_tasks_blocked()
        assert sorted(woken) == [0, 1]

    assert b.broken
    # Waiting on a broken generation raises but wait() stays a checkpoint.
    with assert_checkpoints():
        with pytest.raises(BrokenBarrierError):
            await b.wait()
    assert b.n_waiting == 0


async def test_Barrier_abort_is_idempotent_and_reset_recovers() -> None:
    b = Barrier(2)
    b.abort()
    assert b.broken
    assert b.generation == 0
    b.abort()  # safe to call again with nobody waiting
    assert b.broken
    b.reset()
    assert not b.broken
    assert b.n_waiting == 0
    assert b.generation == 1

    box: list[BarrierWaitResult] = []

    async def one() -> None:
        box.append(await b.wait())

    async with _core.open_nursery() as nursery:
        nursery.start_soon(one)
        nursery.start_soon(one)
    assert b.generation == 2
    assert sum(r.coordinator for r in box) == 1
    assert {r.generation for r in box} == {1}


async def test_Barrier_reset_wakes_waiters_then_new_generation_succeeds() -> None:
    parties = 2
    b = Barrier(parties)
    woken = []

    async def participant() -> None:
        with pytest.raises(BrokenBarrierError):
            await b.wait()
        woken.append(True)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(participant)
        await wait_all_tasks_blocked()
        assert b.n_waiting == 1
        gen_before = b.generation
        b.reset()
        await wait_all_tasks_blocked()
        assert woken == [True]

    assert not b.broken
    assert b.n_waiting == 0
    assert b.generation == gen_before + 1

    # A fresh generation completes normally, proving the abandoned
    # generation's late cleanup cannot corrupt the new one.
    box: list[BarrierWaitResult] = []

    async def pair() -> None:
        box.append(await b.wait())

    async with _core.open_nursery() as nursery:
        nursery.start_soon(pair)
        nursery.start_soon(pair)
    assert b.n_waiting == 0
    assert b.generation == gen_before + 2
    assert sum(r.coordinator for r in box) == 1
    assert {r.generation for r in box} == {gen_before + 1}


async def test_Barrier_cancelled_waiter_breaks_and_wakes_peers() -> None:
    parties = 3
    b = Barrier(parties)
    peer_outcomes: list[str] = []
    admit = [trio.Event() for _ in range(parties)]
    # The cancel scope exists before the final participant enters, so the
    # controller can cancel exactly that participant in the release window.
    last_scope = trio.CancelScope()
    instrument = _CancelLastArriverInstrument(b, last_scope)

    async def peer(index: int) -> None:
        await admit[index].wait()
        try:
            await b.wait()
        except BrokenBarrierError:
            peer_outcomes.append(f"broken-{index}")

    async def last_participant() -> None:
        await admit[parties - 1].wait()
        with last_scope:
            await b.wait()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(peer, 0)
        nursery.start_soon(peer, 1)
        nursery.start_soon(last_participant)

        # First parties-1 participants park; the barrier must not trip yet.
        admit[0].set()
        admit[1].set()
        while b.n_waiting < parties - 1:
            await _core.checkpoint()
        await wait_all_tasks_blocked()
        assert b.n_waiting == parties - 1
        assert not b.broken

        # Admit the final participant and cancel it in its release window.
        instrument.armed = True
        trio.lowlevel.add_instrument(instrument)
        try:
            admit[parties - 1].set()
            await wait_all_tasks_blocked()
        finally:
            trio.lowlevel.remove_instrument(instrument)
        assert instrument.fired

        while len(peer_outcomes) < parties - 1:
            await _core.checkpoint()

    assert sorted(peer_outcomes) == ["broken-0", "broken-1"]
    assert b.broken


async def test_Barrier_statistics_distinguish_waiting_and_generation() -> None:
    b = Barrier(3)
    assert b.parties == 3
    assert b.n_waiting == 0
    assert b.generation == 0
    assert not b.broken

    async def hold() -> None:
        try:
            await b.wait()
        except BrokenBarrierError:
            pass

    async with _core.open_nursery() as nursery:
        nursery.start_soon(hold)
        nursery.start_soon(hold)
        await wait_all_tasks_blocked()
        stats = b.statistics()
        assert stats.tasks_waiting == 2
        assert stats.generation == 0
        assert stats.parties == 3
        assert stats.broken is False
        b.abort()
        await wait_all_tasks_blocked()
        stats = b.statistics()
        assert stats.broken is True
        assert stats.tasks_waiting == 0
        assert stats.generation == 0  # abort leaves the generation number alone

    b.reset()
    stats = b.statistics()
    assert stats.broken is False
    assert stats.tasks_waiting == 0
    assert stats.generation == 1


async def test_Barrier_wait_again_during_release_joins_next_generation() -> None:
    # A non-coordinator task calls wait() again as soon as it wakes, while
    # the generation is being rolled over. It must join the *new* generation
    # rather than be mistaken for a duplicate of the old one, and the two
    # generations must each release exactly once with one coordinator.
    parties = 2
    b = Barrier(parties)
    fast_generations: list[int] = []
    rounds = 3

    async def fast_task() -> None:
        for _ in range(rounds):
            result = await b.wait()
            fast_generations.append(result.generation)

    async def slow_task() -> None:
        for round_index in range(rounds):
            result = await b.wait()
            assert result.generation == round_index
            # Yield a little before re-entering so the fast task can rush
            # ahead into the next generation mid-release.
            await _core.checkpoint()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(fast_task)
        nursery.start_soon(slow_task)

    assert sorted(fast_generations) == [0, 1, 2]
    assert b.generation == rounds
    assert not b.broken
    assert b.n_waiting == 0


async def test_Barrier_abort_before_release_checkpoint_then_reset_then_ok() -> None:
    # abort() while participants are parked breaks the generation; an
    # immediate reset starts a clean generation even though the aborted
    # waiters have not yet processed their wakeups.
    parties = 2
    b = Barrier(parties)
    outcomes: list[str] = []

    async def participant() -> None:
        try:
            await b.wait()
        except BrokenBarrierError:
            outcomes.append("broken")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(participant)
        await wait_all_tasks_blocked()
        assert b.n_waiting == 1
        b.abort()
        # Reset before the parked participant has run; this must start a new
        # generation that the late cleanup of the old one cannot break.
        b.reset()
        await wait_all_tasks_blocked()
        assert outcomes == ["broken"]
        assert not b.broken
        assert b.generation == 1
        assert b.n_waiting == 0


async def test_Barrier_no_reentry_same_task() -> None:
    b = Barrier(2)
    task = _core.current_task()
    # A single coroutine cannot genuinely call wait() twice concurrently,
    # so the guard is exercised directly: pretend this task is already in
    # the generation and confirm wait() refuses it.
    b._gen.waiting.add(task)  # type: ignore[attr-defined]
    try:
        with pytest.raises(RuntimeError, match="can't wait on a Barrier twice"):
            await b.wait()
    finally:
        b._gen.waiting.discard(task)  # type: ignore[attr-defined]
    assert b.n_waiting == 0
