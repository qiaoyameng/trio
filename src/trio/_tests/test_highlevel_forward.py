from __future__ import annotations

import pytest

import trio
from trio.testing import MemorySendStream, memory_stream_pair, wait_all_tasks_blocked

from .. import _core
from .._highlevel_forward import forward_streams
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream


class RecordingStream(trio.abc.HalfCloseableStream):
    """Wraps a stream, recording send_eof/aclose events and counting closes."""

    def __init__(
        self,
        wrapped: trio.abc.HalfCloseableStream,
        events: list[str],
        name: str,
    ) -> None:
        self._wrapped = wrapped
        self._events = events
        self._name = name
        self.close_count = 0

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self._wrapped.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self._wrapped.wait_send_all_might_not_block()

    async def send_eof(self) -> None:
        self._events.append(f"{self._name}-send_eof")
        await self._wrapped.send_eof()

    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        return await self._wrapped.receive_some(max_bytes)

    async def aclose(self) -> None:
        self.close_count += 1
        self._events.append(f"{self._name}-aclose")
        await self._wrapped.aclose()


async def test_forward_streams_half_close() -> None:
    # The upstream peer only generates its response after the request has
    # finished, i.e. after it receives EOF. This only works if the EOF is
    # forwarded as a half-close that leaves the reverse direction open.
    events: list[str] = []
    received = bytearray()
    client, proxy_client_raw = memory_stream_pair()
    proxy_upstream_raw, upstream = memory_stream_pair()
    proxy_client = RecordingStream(proxy_client_raw, events, "proxy-client")
    proxy_upstream = RecordingStream(proxy_upstream_raw, events, "proxy-upstream")

    async def upstream_peer() -> None:
        request = bytearray()
        while True:
            chunk = await upstream.receive_some(1024)
            if not chunk:
                break
            request.extend(chunk)
        events.append("upstream-saw-eof")
        assert bytes(request) == b"ping"
        await upstream.send_all(b"pong")
        events.append("upstream-sent-response")
        await upstream.send_eof()

    async def client_peer() -> None:
        await client.send_all(b"ping")
        await client.send_eof()
        events.append("client-sent-eof")
        while True:
            chunk = await client.receive_some(1024)
            if not chunk:
                break
            received.extend(chunk)
        events.append("client-saw-eof")

    async with trio.open_nursery() as nursery:
        nursery.start_soon(client_peer)
        nursery.start_soon(upstream_peer)
        nursery.start_soon(forward_streams, proxy_client, proxy_upstream)

    # The full response made it back to the client, even though the client
    # had already half-closed its sending side.
    assert bytes(received) == b"pong"
    # The client's EOF was forwarded as a half-close...
    assert events.index("client-sent-eof") < events.index("proxy-upstream-send_eof")
    # ...which is what the upstream saw before responding...
    assert events.index("proxy-upstream-send_eof") < events.index("upstream-saw-eof")
    assert events.index("upstream-saw-eof") < events.index("upstream-sent-response")
    # ...and the response then flowed back over the still-open reverse
    # direction, after the client had sent its EOF.
    assert events.index("client-sent-eof") < events.index("client-saw-eof")
    assert events.index("upstream-sent-response") < events.index("client-saw-eof")
    # Each stream was closed exactly once.
    assert proxy_client.close_count == 1
    assert proxy_upstream.close_count == 1


async def test_forward_streams_send_failure_tears_down_and_closes() -> None:
    events: list[str] = []
    client, proxy_client_raw = memory_stream_pair()
    proxy_upstream_raw, _upstream = memory_stream_pair()
    proxy_client = RecordingStream(proxy_client_raw, events, "proxy-client")
    proxy_upstream = RecordingStream(proxy_upstream_raw, events, "proxy-upstream")

    # Inject a write failure on the proxy -> upstream direction.
    async def broken_send_hook() -> None:
        raise trio.BrokenResourceError("simulated network failure")

    proxy_upstream_raw.send_stream.send_all_hook = broken_send_hook

    # Inject a cleanup failure on the other stream, to check that it does
    # not mask the original forwarding error.
    def broken_close_hook() -> None:
        raise ValueError("simulated cleanup failure")

    proxy_client_raw.send_stream.close_hook = broken_close_hook

    # Memory streams buffer, so the client can finish before forwarding
    # even starts.
    await client.send_all(b"hello")
    await client.send_eof()

    with pytest.RaisesGroup(
        pytest.RaisesExc(trio.BrokenResourceError, match="simulated network failure"),
    ):
        await forward_streams(proxy_client, proxy_upstream)

    # Both streams were reclaimed, exactly once each.
    assert proxy_client.close_count == 1
    assert proxy_upstream.close_count == 1


class GatedSendStream(SendStream):
    """A send stream whose send_all blocks until released."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.send_started = trio.Event()
        self.release_send = trio.Event()
        self.closed = False

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        self.chunks.append(bytes(data))
        self.send_started.set()
        await self.release_send.wait()

    async def wait_send_all_might_not_block(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


class ScriptedReceiveStream(ReceiveStream):
    """A receive stream that returns scripted chunks, then blocks forever."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.calls = 0

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        self.calls += 1
        if self._chunks:
            return self._chunks.pop(0)
        await trio.sleep_forever()

    async def aclose(self) -> None:
        pass


async def test_forward_streams_backpressure() -> None:
    gated_send = GatedSendStream()
    scripted_recv = ScriptedReceiveStream([b"one", b"two", b""])
    left = StapledStream(MemorySendStream(), scripted_recv)
    # The reverse direction EOFs immediately; GatedSendStream is not
    # half-closeable.
    right = StapledStream(gated_send, ScriptedReceiveStream([b""]))

    async with trio.open_nursery() as nursery:
        nursery.start_soon(forward_streams, left, right)
        await gated_send.send_started.wait()
        await wait_all_tasks_blocked()
        # The reader must not run ahead of the blocked writer: backpressure
        # from send_all propagates back to the read side, so at most one
        # chunk is in flight.
        assert scripted_recv.calls == 1
        gated_send.release_send.set()

    assert gated_send.chunks == [b"one", b"two"]
    # `right` is not half-closeable, so the EOF degraded to a full close of
    # the stream's sending side.
    assert gated_send.closed


async def test_forward_streams_cancelled_closes_both_streams() -> None:
    events: list[str] = []
    left, _left_peer = memory_stream_pair()
    right, _right_peer = memory_stream_pair()
    proxy_left = RecordingStream(left, events, "left")
    proxy_right = RecordingStream(right, events, "right")

    async with trio.open_nursery() as nursery:
        nursery.start_soon(forward_streams, proxy_left, proxy_right)
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()

    # External cancellation reclaimed both streams, exactly once each.
    assert proxy_left.close_count == 1
    assert proxy_right.close_count == 1


class HangOnCloseStream(trio.abc.HalfCloseableStream):
    """A stream that EOFs immediately in both directions, but hangs in aclose."""

    def __init__(self) -> None:
        self.close_attempted = False

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        pass

    async def wait_send_all_might_not_block(self) -> None:
        pass

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        return b""

    async def send_eof(self) -> None:
        pass

    async def aclose(self) -> None:
        self.close_attempted = True
        await trio.sleep_forever()


async def test_forward_streams_close_timeout(autojump_clock: _core.MockClock) -> None:
    left = HangOnCloseStream()
    right = HangOnCloseStream()
    # The final closes hang forever, but the close timeout bounds them.
    await forward_streams(left, right, close_timeout=10)
    assert left.close_attempted
    assert right.close_attempted


async def test_forward_streams_max_bytes_must_be_positive() -> None:
    left, right = memory_stream_pair()
    with pytest.raises(ValueError, match="max_bytes"):
        await forward_streams(left, right, max_bytes=0)
