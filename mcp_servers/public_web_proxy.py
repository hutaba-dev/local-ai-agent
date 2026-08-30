"""Small HTTPS CONNECT proxy enforcing public-network-only browser egress."""

from __future__ import annotations

import ipaddress
import select
import socket
import threading
from contextlib import AbstractContextManager
from socketserver import StreamRequestHandler, ThreadingTCPServer


MAX_CONNECTION_BYTES = 16 * 1024 * 1024
MAX_SESSION_BYTES = 64 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 15


def public_addresses(host: str, port: int) -> tuple[tuple[int, tuple[object, ...]], ...]:
    if not host or port != 443:
        raise ValueError("browser proxy permits public HTTPS destinations only")
    addresses: list[tuple[int, tuple[object, ...]]] = []
    for family, socktype, _protocol, _canonical, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        address = ipaddress.ip_address(str(sockaddr[0]))
        if not address.is_global:
            raise ValueError("browser destination resolved to a non-public address")
        candidate = (family, sockaddr)
        if candidate not in addresses:
            addresses.append(candidate)
    if not addresses:
        raise ValueError("browser destination did not resolve")
    return tuple(addresses)


class _ProxyServer(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ConnectHandler)
        self.bytes_relayed = 0
        self.bytes_lock = threading.Lock()

    def reserve(self, amount: int) -> bool:
        with self.bytes_lock:
            if self.bytes_relayed + amount > MAX_SESSION_BYTES:
                return False
            self.bytes_relayed += amount
            return True


class _ConnectHandler(StreamRequestHandler):
    server: _ProxyServer

    def handle(self) -> None:
        self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)
        request_line = self.rfile.readline(4096).decode("ascii", errors="replace").strip()
        while self.rfile.readline(4096) not in {b"\r\n", b"\n", b""}:
            pass
        method, separator, authority = request_line.partition(" ")
        target, _, _version = authority.partition(" ")
        if method != "CONNECT" or not separator:
            self._reject(405)
            return
        host, port = self._authority(target)
        try:
            upstream = self._connect_public(host, port)
        except (OSError, ValueError):
            self._reject(403)
            return
        try:
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()
            self._relay(upstream)
        finally:
            upstream.close()

    @staticmethod
    def _authority(value: str) -> tuple[str, int]:
        if value.startswith("["):
            host, separator, port = value[1:].partition("]:")
        else:
            host, separator, port = value.rpartition(":")
        if not separator or not port.isdigit():
            raise ValueError("invalid CONNECT authority")
        return host, int(port)

    @staticmethod
    def _connect_public(host: str, port: int) -> socket.socket:
        last_error: OSError | None = None
        for family, sockaddr in public_addresses(host, port):
            upstream = socket.socket(family, socket.SOCK_STREAM)
            upstream.settimeout(SOCKET_TIMEOUT_SECONDS)
            try:
                upstream.connect(sockaddr)
                return upstream
            except OSError as error:
                last_error = error
                upstream.close()
        raise last_error or OSError("public destination connection failed")

    def _relay(self, upstream: socket.socket) -> None:
        sockets = (self.connection, upstream)
        connection_bytes = 0
        while connection_bytes < MAX_CONNECTION_BYTES:
            readable, _, exceptional = select.select(sockets, (), sockets, SOCKET_TIMEOUT_SECONDS)
            if exceptional or not readable:
                return
            for source in readable:
                data = source.recv(min(65_536, MAX_CONNECTION_BYTES - connection_bytes))
                if not data or not self.server.reserve(len(data)):
                    return
                connection_bytes += len(data)
                destination = upstream if source is self.connection else self.connection
                destination.sendall(data)

    def _reject(self, status: int) -> None:
        self.wfile.write(f"HTTP/1.1 {status} Forbidden\r\nConnection: close\r\n\r\n".encode("ascii"))


class PublicWebProxy(AbstractContextManager["PublicWebProxy"]):
    def __init__(self) -> None:
        self._server = _ProxyServer()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "PublicWebProxy":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)