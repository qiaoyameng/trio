from __future__ import annotations

import math
import operator
import sys
from functools import partial
from typing import Generic, TypeGuard, TypeVar

import attrs

import trio
from trio._util import final

from .abc import (
    AsyncResource,
    HalfCloseableStream,
    ReceiveStream,
    SendStream,
    Stream,
)

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

SendStreamT = TypeVar("SendStreamT", bound=SendStream)
ReceiveStreamT = TypeVar("ReceiveStreamT", bound=ReceiveStream)
StreamT = TypeVar("StreamT", bound=AsyncResource)


async def aclose_forcefully(resource: AsyncResource) -> None:
    """Close an async resource or async generator immediately, without
    blocking to do any graceful cleanup.

    :class:`~trio.abc.AsyncResource` objects guarantee that if their
    :meth:`~trio.abc.AsyncResource.aclose` method is cancelled, then they will
    still close the resource (albeit in a potentially ungraceful
    fashion). :func:`aclose_forcefully` is a convenience function that
    exploits this behavior to let you force a resource to be closed without
    blocking: it works by calling ``await resource.aclose()`` and then
    cancelling it immediately.

    Most users won't need this, but it may be useful on cleanup paths where
    you can't afford to block, or if you want to close a resource and don't
    care about handling it gracefully. For example, if
    :class:`~trio.SSLStream` encounters an error and cannot perform its
    own graceful close, then there's no point in waiting to gracefully shut
    down the underlying transport either, so it calls ``await
    aclose_forcefully(self.transport_stream)``.

    Note that this function is async, and that it acts as a checkpoint, but
    unlike most async functions it cannot block indefinitely (at least,
    assuming the underlying resource object is correctly implemented).

    """
    with trio.CancelScope() as cs:
        cs.cancel(reason="cancelled during aclose_forcefully")
        await resource.aclose()


def _is_halfclosable(stream: SendStream) -> TypeGuard[HalfCloseableStream]:
    """Check if the stream has a send_eof() method."""
    return hasattr(stream, "send_eof")


async def _close_stream(
    stream: AsyncResource,
    *,
    deadline: float,
) -> None:
    if deadline == math.inf:
        await stream.aclose()
        return

    with trio.move_on_at(deadline) as cancel_scope:
        await stream.aclose()
    if cancel_scope.cancelled_caught:
        await aclose_forcefully(stream)


async def _close_streams(
    streams: tuple[StreamT, StreamT],
    *,
    close_timeout: float | None,
) -> list[BaseException]:
    deadline = math.inf
    if close_timeout is not None and close_timeout != math.inf:
        deadline = trio.current_time() + close_timeout
    errors: list[BaseException] = []
    unique_streams: list[StreamT] = []
    seen_streams: set[int] = set()
    for stream in streams:
        stream_id = id(stream)
        if stream_id not in seen_streams:
            seen_streams.add(stream_id)
            unique_streams.append(stream)

    try:
        async with trio.open_nursery() as nursery:
            for stream in unique_streams:
                nursery.start_soon(
                    partial(
                        _close_stream,
                        stream,
                        deadline=deadline,
                    ),
                )
    except BaseException as error:
        _collect_errors(error, errors)

    return errors


def _collect_errors(error: BaseException, errors: list[BaseException]) -> None:
    if isinstance(error, BaseExceptionGroup):
        for nested_error in error.exceptions:
            _collect_errors(nested_error, errors)
    else:
        errors.append(error)


def _raise_with_cleanup_errors(
    original_error: BaseException,
    cleanup_errors: list[BaseException],
) -> None:
    if cleanup_errors:
        cleanup_group = BaseExceptionGroup(
            "errors while closing streams",
            cleanup_errors,
        )
        if original_error.__context__ is not None:
            cleanup_group.__context__ = original_error.__context__
        original_error.__context__ = cleanup_group
    raise original_error


async def forward_stream(
    stream1: Stream,
    stream2: Stream,
    *,
    max_bytes: int = 65536,
    close_timeout: float | None = None,
) -> None:
    """Forward data between two bidirectional streams until both directions end.

    When one peer sends an end-of-file marker, only the sending direction of
    the other peer is closed. The reverse direction continues forwarding until
    it also reaches end-of-file. Streams that do not implement
    :class:`~trio.abc.HalfCloseableStream` cannot represent this state, so an
    end-of-file in either direction causes both streams to be closed
    immediately and this coroutine returns.

    At most ``max_bytes`` bytes are buffered in each direction. A blocked
    :meth:`~trio.abc.SendStream.send_all` therefore stops reading from the
    source stream until the destination can make progress.

    Any forwarding error cancels the other direction and closes both streams.
    External cancellation does the same. If ``close_timeout`` is set, it limits
    the total time spent waiting for graceful stream closure; streams that
    remain unresponsive are force-closed. By default, or if it is ``None`` or
    infinity, graceful closure is allowed to wait indefinitely unless the
    forwarding operation is externally cancelled; external cancellation always
    switches immediately to force closure.

    The first transfer error is raised. Errors encountered while closing are
    attached to it rather than replacing it.
    """
    max_bytes = operator.index(max_bytes)
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    if close_timeout is not None:
        close_timeout = float(close_timeout)
        if close_timeout < 0 or math.isnan(close_timeout):
            raise ValueError("close_timeout must be non-negative or None")

    streams = (stream1, stream2)
    transport_error: BaseException | None = None
    internal_shutdown = False
    transport_cancelled = False

    def downgrade() -> None:
        nonlocal internal_shutdown
        internal_shutdown = True
        forward_scope.cancel(reason="stream does not support half-closing")

    async def relay(source: ReceiveStream, destination: SendStream) -> None:
        nonlocal transport_cancelled, transport_error

        try:
            while True:
                data = await source.receive_some(max_bytes)
                if data:
                    await destination.send_all(data)
                    continue

                if _is_halfclosable(destination):
                    await destination.send_eof()
                    return

                downgrade()
                await trio.sleep(0)
                return
        except trio.Cancelled:
            raise
        except Exception as error:
            if transport_error is None and not forward_scope.cancel_called:
                transport_error = error
                transport_cancelled = True
                forward_scope.cancel(reason="error while forwarding stream data")
            return

    try:
        with trio.CancelScope() as forward_scope:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(relay, stream1, stream2)
                nursery.start_soon(relay, stream2, stream1)
    except BaseException as error:
        if transport_error is None:
            transport_error = error

    transfer_errors: list[BaseException] = []
    if transport_error is not None:
        _collect_errors(transport_error, transfer_errors)

    effective_close_timeout = close_timeout
    external_cancellation = (
        forward_scope.cancelled_caught
        and not internal_shutdown
        and not transport_cancelled
    )
    if external_cancellation and close_timeout is None:
        effective_close_timeout = 0

    cleanup_errors = await _close_streams(
        streams,
        close_timeout=effective_close_timeout,
    )

    if external_cancellation:
        await trio.lowlevel.checkpoint_if_cancelled()

    if transfer_errors:
        first_error, *remaining_errors = transfer_errors
        cleanup_errors[0:0] = remaining_errors
        _raise_with_cleanup_errors(first_error, cleanup_errors)
    elif cleanup_errors:
        raise BaseExceptionGroup(
            "errors while closing streams",
            cleanup_errors,
        )

forward_streams = forward_stream


@final
@attrs.define(eq=False, slots=False)
class StapledStream(
    HalfCloseableStream,
    Generic[SendStreamT, ReceiveStreamT],
):
    """This class `staples <https://en.wikipedia.org/wiki/Staple_(fastener)>`__
    together two unidirectional streams to make single bidirectional stream.

    Args:
      send_stream (~trio.abc.SendStream): The stream to use for sending.
      receive_stream (~trio.abc.ReceiveStream): The stream to use for
          receiving.

    Example:

       A silly way to make a stream that echoes back whatever you write to
       it::

          left, right = trio.testing.memory_stream_pair()
          echo_stream = StapledStream(SocketStream(left), SocketStream(right))
          await echo_stream.send_all(b"x")
          assert await echo_stream.receive_some() == b"x"

    :class:`StapledStream` objects implement the methods in the
    :class:`~trio.abc.HalfCloseableStream` interface. They also have two
    additional public attributes:

    .. attribute:: send_stream

       The underlying :class:`~trio.abc.SendStream`. :meth:`send_all` and
       :meth:`wait_send_all_might_not_block` are delegated to this object.

    .. attribute:: receive_stream

       The underlying :class:`~trio.abc.ReceiveStream`. :meth:`receive_some`
       is delegated to this object.

    """

    send_stream: SendStreamT
    receive_stream: ReceiveStreamT

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Calls ``self.send_stream.send_all``."""
        return await self.send_stream.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        """Calls ``self.send_stream.wait_send_all_might_not_block``."""
        return await self.send_stream.wait_send_all_might_not_block()

    async def send_eof(self) -> None:
        """Shuts down the send side of the stream.

        If :meth:`self.send_stream.send_eof() <trio.abc.HalfCloseableStream.send_eof>` exists,
        then this calls it. Otherwise, this calls
        :meth:`self.send_stream.aclose() <trio.abc.AsyncResource.aclose>`.
        """
        stream = self.send_stream
        if _is_halfclosable(stream):
            return await stream.send_eof()
        else:
            return await stream.aclose()

    # we intentionally accept more types from the caller than we support returning
    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        """Calls ``self.receive_stream.receive_some``."""
        return await self.receive_stream.receive_some(max_bytes)

    async def aclose(self) -> None:
        """Calls ``aclose`` on both underlying streams."""
        try:
            await self.send_stream.aclose()
        finally:
            await self.receive_stream.aclose()
