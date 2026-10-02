from __future__ import annotations

import sys
from functools import partial
from typing import NoReturn

import attrs
import pytest

import trio

from .._highlevel_generic import StapledStream, _close_streams, forward_stream
from ..abc import HalfCloseableStream, ReceiveStream, SendStream, Stream
from ..testing import memory_stream_one_way_pair, wait_all_tasks_blocked

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


@attrs.define(slots=False)
class RecordSendStream(SendStream):
    record: list[str | tuple[str, object]] = attrs.Factory(list)

    async def send_all(self, data: object) -> None:
        self.record.append(("send_all", data))

    async def wait_send_all_might_not_block(self) -> None:
        self.record.append("wait_send_all_might_not_block")

    async def aclose(self) -> None:
        self.record.append("aclose")


@attrs.define(slots=False)
class RecordReceiveStream(ReceiveStream):
    record: list[str | tuple[str, int | None]] = attrs.Factory(list)

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        self.record.append(("receive_some", max_bytes))
        return b""

    async def aclose(self) -> None:
        self.record.append("aclose")


async def test_StapledStream() -> None:
    send_stream = RecordSendStream()
    receive_stream = RecordReceiveStream()
    stapled = StapledStream(send_stream, receive_stream)

    assert stapled.send_stream is send_stream
    assert stapled.receive_stream is receive_stream

    await stapled.send_all(b"foo")
    await stapled.wait_send_all_might_not_block()
    assert send_stream.record == [
        ("send_all", b"foo"),
        "wait_send_all_might_not_block",
    ]
    send_stream.record.clear()

    await stapled.send_eof()
    assert send_stream.record == ["aclose"]
    send_stream.record.clear()

    async def fake_send_eof() -> None:
        send_stream.record.append("send_eof")

    send_stream.send_eof = fake_send_eof  # type: ignore[attr-defined]
    await stapled.send_eof()
    assert send_stream.record == ["send_eof"]

    send_stream.record.clear()
    assert receive_stream.record == []

    await stapled.receive_some(1234)
    assert receive_stream.record == [("receive_some", 1234)]
    assert send_stream.record == []
    receive_stream.record.clear()

    await stapled.aclose()
    assert receive_stream.record == ["aclose"]
    assert send_stream.record == ["aclose"]


async def test_StapledStream_with_erroring_close() -> None:
    # Make sure that if one of the aclose methods errors out, then the other
    # one still gets called.
    class BrokenSendStream(RecordSendStream):
        async def aclose(self) -> NoReturn:
            await super().aclose()
            raise ValueError("send error")

    class BrokenReceiveStream(RecordReceiveStream):
        async def aclose(self) -> NoReturn:
            await super().aclose()
            raise ValueError("recv error")

    stapled = StapledStream(BrokenSendStream(), BrokenReceiveStream())

    with pytest.raises(ValueError, match=r"^(send|recv) error$") as excinfo:
        await stapled.aclose()
    assert isinstance(excinfo.value.__context__, ValueError)

    assert stapled.send_stream.record == ["aclose"]
    assert stapled.receive_stream.record == ["aclose"]


@attrs.define(eq=False, slots=False)
class CountingSendStream(SendStream):
    wrapped: SendStream
    events: list[str]

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self.wrapped.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self.wrapped.wait_send_all_might_not_block()

    async def send_eof(self) -> None:
        self.events.append("send_eof")
        self.wrapped.close()

    async def aclose(self) -> None:
        self.events.append("aclose-send")
        await self.wrapped.aclose()


@attrs.define(eq=False, slots=False)
class CountingReceiveStream(ReceiveStream):
    wrapped: ReceiveStream
    events: list[str]

    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        return await self.wrapped.receive_some(max_bytes)

    async def aclose(self) -> None:
        self.events.append("aclose-receive")
        await self.wrapped.aclose()


@attrs.define(eq=False, slots=False)
class CountingHalfCloseableStream(HalfCloseableStream):
    wrapped: HalfCloseableStream

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self.wrapped.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self.wrapped.wait_send_all_might_not_block()

    async def send_eof(self) -> None:
        await self.wrapped.send_eof()

    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        return await self.wrapped.receive_some(max_bytes)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


@attrs.define(eq=False, slots=False)
class CountingStream(Stream):
    wrapped: HalfCloseableStream

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self.wrapped.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self.wrapped.wait_send_all_might_not_block()

    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        return await self.wrapped.receive_some(max_bytes)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


def _staple(send: SendStream, receive: ReceiveStream) -> StapledStream:
    return StapledStream(send, receive)


def _proxy_streams(
    *, half_closeable: bool = True,
) -> tuple[Stream, Stream, StapledStream, StapledStream, list[str], list[str]]:
    client_request_send, client_request_recv = memory_stream_one_way_pair()
    client_response_send, client_response_recv = memory_stream_one_way_pair()
    upstream_request_send, upstream_request_recv = memory_stream_one_way_pair()
    upstream_response_send, upstream_response_recv = memory_stream_one_way_pair()

    client = _staple(client_request_send, client_response_recv)
    upstream = _staple(upstream_response_send, upstream_request_recv)

    client_events: list[str] = []
    upstream_events: list[str] = []
    client_send = CountingSendStream(client_response_send, client_events)
    client_receive = CountingReceiveStream(client_request_recv, client_events)
    upstream_send = CountingSendStream(upstream_request_send, upstream_events)
    upstream_receive = CountingReceiveStream(
        upstream_response_recv,
        upstream_events,
    )
    client_proxy = _staple(client_send, client_receive)
    upstream_proxy = _staple(upstream_send, upstream_receive)

    return (
        (
            CountingHalfCloseableStream(client_proxy)
            if half_closeable
            else CountingStream(client_proxy)
        ),
        (
            CountingHalfCloseableStream(upstream_proxy)
            if half_closeable
            else CountingStream(upstream_proxy)
        ),
        client,
        upstream,
        client_events,
        upstream_events,
    )


async def test_forward_stream_half_closes_before_final_response() -> None:
    (
        client_stream,
        upstream_stream,
        client,
        upstream,
        client_events,
        upstream_events,
    ) = _proxy_streams()

    async def server() -> None:
        request = bytearray()
        while True:
            chunk = await upstream.receive_some(3)
            if not chunk:
                break
            request += chunk
        await upstream.send_all(b"response:" + request)
        await upstream.send_eof()

    async with trio.open_nursery() as nursery:
        nursery.start_soon(forward_stream, client_stream, upstream_stream)
        nursery.start_soon(server)

        await client.send_all(b"abc")
        await client.send_eof()
        await wait_all_tasks_blocked()
        assert await client.receive_some() == b"response:abc"

    assert client_events == [
        "send_eof",
        "aclose-send",
        "aclose-receive",
    ]
    assert upstream_events == [
        "send_eof",
        "aclose-send",
        "aclose-receive",
    ]
    assert upstream_events.index("send_eof") < upstream_events.index("aclose-send")


async def test_forward_stream_preserves_write_error_and_closes_once() -> None:
    (
        client_stream,
        upstream_stream,
        client,
        upstream,
        client_events,
        upstream_events,
    ) = _proxy_streams()

    async def write_failure() -> None:
        raise OSError("write failed")

    original_send_all = client_stream.send_all
    failed = False

    async def fail_once(data: bytes | bytearray | memoryview) -> None:
        nonlocal failed
        if failed:
            await original_send_all(data)
        else:
            failed = True
            await original_send_all(data)
            await write_failure()

    client_stream.send_all = fail_once  # type: ignore[method-assign]

    async def server() -> None:
        request = bytearray()
        while True:
            chunk = await upstream.receive_some(3)
            if not chunk:
                break
            request += chunk
        assert request == b"request"
        try:
            await upstream.send_all(b"response")
        except (trio.BrokenResourceError, trio.ClosedResourceError):
            pass

    with pytest.raises(ExceptionGroup) as excinfo:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(forward_stream, client_stream, upstream_stream)
            nursery.start_soon(server)
            await client.send_all(b"request")
            await client.send_eof()
            await wait_all_tasks_blocked()

    write_error = excinfo.value.exceptions[0]
    assert isinstance(write_error, OSError)
    assert str(write_error) == "write failed"
    assert failed
    assert await client.receive_some() == b"response"
    assert client_events == [
        "aclose-send",
        "aclose-receive",
    ]
    assert upstream_events == [
        "send_eof",
        "aclose-send",
        "aclose-receive",
    ]


async def test_forward_stream_downgrades_without_half_close() -> None:
    (
        client_stream,
        upstream_stream,
        client,
        upstream,
        client_events,
        upstream_events,
    ) = _proxy_streams(
        half_closeable=False,
    )

    async def server() -> None:
        assert await upstream.receive_some() == b"request"

    async with trio.open_nursery() as nursery:
        nursery.start_soon(forward_stream, client_stream, upstream_stream)
        nursery.start_soon(server)
        await client.send_all(b"request")
        await client.send_eof()
        await wait_all_tasks_blocked()

    assert client_events == [
        "aclose-send",
        "aclose-receive",
    ]
    assert upstream_events == [
        "aclose-send",
        "aclose-receive",
    ]


async def test_forward_stream_cancels_and_closes_both_streams() -> None:
    a_to_b_send, a_to_b_recv = memory_stream_one_way_pair()
    b_to_a_send, b_to_a_recv = memory_stream_one_way_pair()
    a_events: list[str] = []
    b_events: list[str] = []
    a = CountingHalfCloseableStream(
        _staple(
            CountingSendStream(b_to_a_send, a_events),
            CountingReceiveStream(a_to_b_recv, a_events),
        ),
    )
    b = CountingHalfCloseableStream(
        _staple(
            CountingSendStream(a_to_b_send, b_events),
            CountingReceiveStream(b_to_a_recv, b_events),
        ),
    )

    cancelled_during_forward = False

    async def cancel_after_idle(scope: trio.CancelScope) -> None:
        nonlocal cancelled_during_forward
        await wait_all_tasks_blocked()
        cancelled_during_forward = True
        scope.cancel()

    with trio.move_on_after(0.1) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(forward_stream, a, b)
            nursery.start_soon(cancel_after_idle, cancel_scope)

    assert cancelled_during_forward
    assert cancel_scope.cancelled_caught
    assert a_events == ["aclose-send", "aclose-receive"]
    assert b_events == ["aclose-send", "aclose-receive"]


@attrs.define(eq=False, slots=False)
class SlowClosingSendStream(CountingSendStream):
    force_closed = False
    close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.events.append("aclose-send")
        if self.close_calls == 2:
            self.force_closed = True
            await self.wrapped.aclose()
            return
        await trio.lowlevel.checkpoint_if_cancelled()
        await self.wrapped.aclose()


async def test_forward_stream_force_closes_after_deadline() -> None:
    a_to_b_send, a_to_b_recv = memory_stream_one_way_pair()
    b_to_a_send, b_to_a_recv = memory_stream_one_way_pair()
    a_events: list[str] = []
    b_events: list[str] = []
    a = CountingHalfCloseableStream(
        _staple(
            SlowClosingSendStream(b_to_a_send, a_events),
            CountingReceiveStream(a_to_b_recv, a_events),
        ),
    )
    b = CountingHalfCloseableStream(
        _staple(
            SlowClosingSendStream(a_to_b_send, b_events),
            CountingReceiveStream(b_to_a_recv, b_events),
        ),
    )

    await _close_streams((a, b), close_timeout=0)

    assert a_events.count("aclose-send") == 2
    assert b_events.count("aclose-send") == 2
    assert a_events.count("aclose-receive") == 2
    assert b_events.count("aclose-receive") == 2
    assert isinstance(a.wrapped.send_stream, SlowClosingSendStream)
    assert isinstance(b.wrapped.send_stream, SlowClosingSendStream)
    assert a.wrapped.send_stream.close_calls == 2
    assert b.wrapped.send_stream.close_calls == 2
    assert a.wrapped.send_stream.force_closed
    assert b.wrapped.send_stream.force_closed


async def test_forward_stream_backpressure_stops_reading() -> None:
    send_unblocked = trio.Event()
    send_started = trio.Event()
    second_read_happened = trio.Event()

    class BlockingSendStream(SendStream):
        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            assert data == b"chunk"
            send_started.set()
            await send_unblocked.wait()

        async def wait_send_all_might_not_block(self) -> None:
            await trio.lowlevel.checkpoint()

        async def aclose(self) -> None:
            send_unblocked.set()

    class ImmediateEOFReceiveStream(ReceiveStream):
        async def receive_some(self, max_bytes: int | None = None) -> bytes:
            return b""

        async def aclose(self) -> None:
            pass

    class ScriptedStream(HalfCloseableStream):
        read_count = 0

        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            raise AssertionError("unexpected send")

        async def wait_send_all_might_not_block(self) -> None:
            await trio.lowlevel.checkpoint()

        async def send_eof(self) -> None:
            pass

        async def receive_some(self, max_bytes: int | None = None) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"chunk"
            second_read_happened.set()
            return b""

        async def aclose(self) -> None:
            send_unblocked.set()

    a = ScriptedStream()
    b = _staple(BlockingSendStream(), ImmediateEOFReceiveStream())

    async with trio.open_nursery() as nursery:
        nursery.start_soon(partial(forward_stream, a, b, max_bytes=5))
        await send_started.wait()
        assert a.read_count == 1
        send_unblocked.set()

    assert second_read_happened.is_set()


@attrs.define(eq=False, slots=False)
class CloseFailingSendStream(CountingSendStream):
    async def aclose(self) -> NoReturn:
        await super().aclose()
        raise ValueError("close failed")


async def test_forward_stream_close_error_keeps_transfer_error() -> None:
    (
        client_stream,
        upstream_stream,
        client,
        upstream,
        client_events,
        _,
    ) = _proxy_streams()
    stapled_stream = client_stream.wrapped
    assert isinstance(stapled_stream, StapledStream)
    assert isinstance(stapled_stream.send_stream, CountingSendStream)
    assert isinstance(stapled_stream.receive_stream, CountingReceiveStream)
    stapled_stream.send_stream = CloseFailingSendStream(
        stapled_stream.send_stream.wrapped,
        client_events,
    )

    original_send_all = upstream_stream.send_all

    async def fail_once(data: bytes | bytearray | memoryview) -> None:
        await original_send_all(data)
        raise OSError("write failed")

    upstream_stream.send_all = fail_once  # type: ignore[method-assign]

    async def server() -> None:
        assert await upstream.receive_some() == b"request"

    with pytest.raises(ExceptionGroup) as excinfo:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(forward_stream, client_stream, upstream_stream)
            nursery.start_soon(server)
            await client.send_all(b"request")
            await wait_all_tasks_blocked()

    first_error = excinfo.value.exceptions[0]
    assert isinstance(first_error, OSError)
    assert str(first_error) == "write failed"
    assert isinstance(first_error.__context__, ExceptionGroup)
    close_error = first_error.__context__.exceptions[0]
    assert isinstance(close_error, ValueError)
    assert str(close_error) == "close failed"
