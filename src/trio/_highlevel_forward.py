"""Bidirectional stream forwarding, with half-close support."""

from __future__ import annotations

import trio

from ._highlevel_generic import _is_halfclosable
from .abc import Stream

#: Default value of the ``max_bytes`` argument to :func:`forward_streams`.
DEFAULT_FORWARD_BUFFER_SIZE = 65536


async def forward_streams(
    left: Stream,
    right: Stream,
    *,
    max_bytes: int = DEFAULT_FORWARD_BUFFER_SIZE,
    close_timeout: float | None = None,
) -> None:
    """Forward data between two streams, in both directions, until both
    directions have finished.

    This is the building block you want when writing things like TCP
    proxies, where a naive "copy bytes back and forth" loop gets the
    details subtly wrong. For example, once a client has sent its request
    it typically half-closes its sending side, and the server only sends
    its final response *after* seeing that EOF -- so simply closing both
    streams as soon as one direction reaches EOF would truncate the
    response. `forward_streams` handles this correctly.

    Args:
      left (~trio.abc.Stream): The first stream.
      right (~trio.abc.Stream): The second stream.
      max_bytes (int): The maximum number of bytes to read in a single
          :meth:`~trio.abc.ReceiveStream.receive_some` call. This also
          bounds the amount of data buffered in each direction: data is
          only read from a stream once the previous chunk has been fully
          written to the other stream, so a blocked
          :meth:`~trio.abc.SendStream.send_all` applies backpressure to
          the corresponding reader. At most ``max_bytes`` bytes are in
          flight per direction.
      close_timeout (float or None): The number of seconds to wait for each
          stream's :meth:`~trio.abc.AsyncResource.aclose` during final
          cleanup before giving up on it (like
          :func:`trio.move_on_after`, no error is raised if the timeout
          expires). If None (the default), wait indefinitely.

    Half-close semantics:

      When one side reaches EOF, only the *sending* side of the other
      stream is shut down (using
      :meth:`~trio.abc.HalfCloseableStream.send_eof`), and forwarding in
      the reverse direction continues until it too reaches EOF. If the
      other stream does not support :meth:`~trio.abc.HalfCloseableStream.send_eof`,
      the EOF cannot be propagated on its own, so as a fallback the whole
      stream is closed with :meth:`~trio.abc.AsyncResource.aclose` --
      which, as a side effect, also shuts down the reverse direction.

    Error handling and cleanup:

      If forwarding in either direction raises an error, the other
      direction is cancelled and both streams are closed before the error
      propagates (wrapped in an :exc:`ExceptionGroup`, following the usual
      nursery semantics). Cancelling the call to `forward_streams` also
      closes both streams. If cleanup itself fails while a forwarding
      error is already propagating, the cleanup error is suppressed so
      that it cannot hide the original error.

    Example -- a minimal TCP proxy::

        async def handle_connection(client_stream):
            upstream = await trio.open_tcp_stream("example.com", 80)
            await trio.forward_streams(client_stream, upstream)

    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    async def forward_one_way(source: Stream, destination: Stream) -> None:
        while True:
            data = await source.receive_some(max_bytes)
            if not data:
                if _is_halfclosable(destination):
                    # Half-close: the reverse direction keeps flowing.
                    await destination.send_eof()
                else:
                    # The destination cannot half-close, so the only way
                    # to propagate the EOF is to close it entirely (which
                    # also tears down the reverse direction).
                    await destination.aclose()
                return
            await destination.send_all(data)

    pending_error: BaseException | None = None
    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                forward_one_way,
                left,
                right,
                name=f"trio.forward_streams: {left!r} -> {right!r}",
            )
            nursery.start_soon(
                forward_one_way,
                right,
                left,
                name=f"trio.forward_streams: {right!r} -> {left!r}",
            )
    except BaseException as error:
        pending_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        for stream in (left, right):
            try:
                if close_timeout is None:
                    await stream.aclose()
                else:
                    with trio.move_on_after(close_timeout):
                        await stream.aclose()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        # A cleanup failure must never hide the error that tore down the
        # forwarding in the first place.
        if cleanup_error is not None and pending_error is None:
            raise cleanup_error
