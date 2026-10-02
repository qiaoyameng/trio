"""Task handles: opt-in access to a nursery child task's result.

Unlike an asyncio Future, a :class:`TaskHandle` does not own the task it
refers to. The task is still spawned in, and supervised by, a nursery. The
handle only provides a way to wait for that one task's result without
leaving the nursery, and to request cancellation of that one task. The
nursery keeps applying its normal exception rules when it closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import attrs
from outcome import Error, Outcome, Value

from . import _core
from ._util import NoPublicConstructor, final

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from ._core import CancelScope, Task

RetT = TypeVar("RetT")


async def _run_task_with_handle(
    coro: Coroutine[object, object, RetT],
    handle: TaskHandle[RetT],
) -> None:
    # The per-task cancel scope makes handle.cancel() cancel just this one
    # child task; the nursery's own scope still surrounds it, so cancelling
    # the nursery as a whole keeps working exactly as before.
    task_outcome: Outcome[RetT] | None = None
    with handle._cancel_scope:
        try:
            value: RetT = await coro
        except BaseException as exc:
            task_outcome = Error(exc)
        else:
            task_outcome = Value(value)
        if isinstance(task_outcome, Error) and isinstance(
            task_outcome.error,
            _core.Cancelled,
        ):
            # A Cancelled that reached the user coroutine was delivered by
            # a cancelled scope. Let the surrounding scope (the handle's own
            # for handle.cancel(), or an outer one such as the nursery's)
            # filter it in the normal way: re-raise it *inside* the `with`,
            # where that scope can catch it. Waiters still need to be woken.
            handle._set_outcome(Value(None))
            raise task_outcome.error
        # A real return value or a real failure: publish it, and let failures
        # propagate (still inside the `with`, so scope mechanics apply).
        handle._set_outcome(task_outcome)
        if isinstance(task_outcome, Error):
            raise task_outcome.error
    assert task_outcome is not None


@final
@attrs.define(eq=False, repr=False)
class TaskHandle(Generic[RetT], metaclass=NoPublicConstructor):
    """A handle to a single task started in a nursery.

    Returned by
    :meth:`Nursery.start_soon_with_handle <trio.Nursery.start_soon_with_handle>`.
    The handle does not keep the referenced task alive; the nursery that
    spawned the task remains responsible for waiting for it. It only lets
    you observe the task's outcome and request cancellation of just that
    task, without turning the nursery into a pool of detached background
    tasks.

    Any number of tasks can concurrently wait on one handle; they all see
    the same result object. Cancelling a task that is waiting on the handle
    only cancels that wait -- it never cancels the referenced task. If the
    referenced task fails, waiting on the handle re-raises the failure, but
    the nursery still collects that failure and raises it on exit in the
    usual way.

    Attributes:
        cancel_scope: The :class:`~trio.CancelScope` that directly surrounds
            the referenced task. Cancelling it (or calling
            :meth:`cancel`) requests cancellation of just this one task.
    """

    _cancel_scope: CancelScope
    _parking_lot: _core.ParkingLot = attrs.field(
        init=False,
        factory=_core.ParkingLot,
    )
    _outcome: Outcome[RetT] | None = attrs.field(default=None, init=False)
    _task: Task | None = attrs.field(default=None, init=False)

    def __repr__(self) -> str:
        status = "complete" if self._outcome is not None else "pending"
        return f"<TaskHandle for {self._task!r}, {status}>"

    @property
    def cancel_scope(self) -> CancelScope:
        return self._cancel_scope

    @property
    def task(self) -> Task | None:
        """The low-level :class:`~trio.lowlevel.Task` this handle refers to.

        This is ``None`` for the brief window after the handle is returned
        but before the scheduler first runs the new task.
        """
        return self._task

    @property
    def done(self) -> bool:
        """`bool`: True once the referenced task has finished."""
        return self._outcome is not None

    @property
    def outcome(self) -> Outcome[RetT]:
        """The :class:`outcome.Outcome` of the referenced task.

        Raises:
            WouldBlock: if the task has not finished yet. Use
                :meth:`wait` or :meth:`join` to wait for it.
        """
        if self._outcome is None:
            raise _core.WouldBlock
        return self._outcome

    def _set_outcome(self, task_outcome: Outcome[RetT]) -> None:
        if self._outcome is not None:  # pragma: no cover
            return
        self._outcome = task_outcome
        self._parking_lot.unpark_all()

    async def join(self) -> None:
        """Wait for the referenced task to finish.

        Unlike :meth:`wait`, this never re-raises the task's exception;
        use the :attr:`outcome` attribute afterwards to inspect it. A
        :exc:`~trio.Cancelled` raised here cancels only the joining task,
        never the referenced task, and the result is shared with every
        other waiter.
        """
        if self._outcome is None:
            await self._parking_lot.park()

    async def wait(self) -> RetT:
        """Wait for the referenced task to finish and return its result.

        If the task raised an exception, then the same exception object is
        re-raised here for every caller. The nursery still receives that
        exception and applies its normal rules when it closes, so reading
        a failure through the handle does not suppress it.

        Cancelling this wait only cancels the wait itself; the referenced
        task keeps running.
        """
        await self.join()
        # Don't use Outcome.unwrap(): outcome objects are one-shot, while
        # any number of waiters share this stored outcome.
        task_outcome = self._outcome
        assert task_outcome is not None
        if isinstance(task_outcome, Error):
            raise task_outcome.error
        return task_outcome.value

    def cancel(self, reason: str | None = None) -> None:
        """Request cancellation of just the referenced task.

        This cancels the handle's :attr:`cancel_scope`, which directly
        surrounds only that one child task. Other children of the same
        nursery are unaffected. It is idempotent and safe to call before
        the task has started or after it has finished.
        """
        self._cancel_scope.cancel(reason)


# Exposed for Nursery.start_soon_with_handle (avoids an import cycle).
_create_task_handle = TaskHandle._create
