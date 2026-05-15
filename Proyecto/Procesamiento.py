from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import struct
import pandas as pd
from scapy.all import RawPcapReader


"""
Salida exacta:
 ts,src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state,missed_bytes,src_pkts,src_ip_bytes,dst_pkts,dst_ip_bytes,dns_query,dns_qclass,dns_qtype,dns_rcode,dns_AA,dns_RD,dns_RA,dns_rejected,ssl_version,ssl_cipher,ssl_resumed,ssl_established,ssl_subject,ssl_issuer,http_trans_depth,http_method,http_uri,http_referrer,http_version,http_request_body_len,http_response_body_len,http_status_code,http_user_agent,http_orig_mime_types,http_resp_mime_types,weird_name,weird_addl,weird_notice,label,type
"""

ATTACK_FOLDER_MAP = {
    "injection_normal": "injection",
    "mitm_normal": "mitm",
    "normal_backdoor": "backdoor",
    "normal_ddos": "ddos",
}

OUTPUT_COLUMNS = [
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service",
    "duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_referrer", "http_version",
    "http_request_body_len", "http_response_body_len", "http_status_code", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice",
    "label", "type",
]

SERVICE_PORT_MAP = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3", 123: "ntp", 135: "msrpc",
    137: "netbios-ns", 138: "netbios-dgm", 139: "netbios-ssn", 143: "imap", 161: "snmp",
    389: "ldap", 443: "https", 445: "smb", 502: "modbus", 554: "rtsp", 631: "ipp",
    993: "imaps", 995: "pop3s", 1883: "mqtt", 20000: "dnp3", 2404: "iec104",
    44818: "ethernet-ip", 47808: "bacnet", 5683: "coap",
}

ETH_P_IP = 0x0800
ETH_P_IPV6 = 0x86DD
IP_PROTO_ICMP = 1
IP_PROTO_TCP = 6
IP_PROTO_UDP = 17

HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ", b"OPTIONS ", b"PATCH ", b"TRACE ", b"CONNECT ")


@dataclass(slots=True)
class FlowRecord:
    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    service: str
    label: str
    type: str
    start_ts: float
    end_ts: float
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    src_ip_bytes: int = 0
    dst_ip_bytes: int = 0
    missed_bytes: int = 0
    tcp_syn: int = 0
    tcp_ack: int = 0
    tcp_fin: int = 0
    tcp_rst: int = 0
    tcp_psh: int = 0
    tcp_urg: int = 0
    dns_query: str = ""
    dns_qclass: str = ""
    dns_qtype: str = ""
    dns_rcode: str = ""
    dns_AA: int = 0
    dns_RD: int = 0
    dns_RA: int = 0
    dns_rejected: int = 0
    ssl_version: str = ""
    ssl_cipher: str = ""
    ssl_resumed: int = 0
    ssl_established: int = 0
    ssl_subject: str = ""
    ssl_issuer: str = ""
    http_trans_depth: int = 0
    http_method: str = ""
    http_uri: str = ""
    http_referrer: str = ""
    http_version: str = ""
    http_request_body_len: int = 0
    http_response_body_len: int = 0
    http_status_code: str = ""
    http_user_agent: str = ""
    http_orig_mime_types: str = ""
    http_resp_mime_types: str = ""
    weird_name: str = ""
    weird_addl: str = ""
    weird_notice: str = ""

    def update_time(self, ts: float) -> None:
        if ts < self.start_ts:
            self.start_ts = ts
        if ts > self.end_ts:
            self.end_ts = ts
        self.ts = self.start_ts

    def update_direction(self, direction: str, payload_len: int, ip_len: int, tcp_flags: int = 0) -> None:
        if direction == "src_to_dst":
            self.src_pkts += 1
            self.src_bytes += payload_len
            self.src_ip_bytes += ip_len
        else:
            self.dst_pkts += 1
            self.dst_bytes += payload_len
            self.dst_ip_bytes += ip_len

        if tcp_flags:
            self.tcp_fin += int(bool(tcp_flags & 0x01))
            self.tcp_syn += int(bool(tcp_flags & 0x02))
            self.tcp_rst += int(bool(tcp_flags & 0x04))
            self.tcp_psh += int(bool(tcp_flags & 0x08))
            self.tcp_ack += int(bool(tcp_flags & 0x10))
            self.tcp_urg += int(bool(tcp_flags & 0x20))

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    def conn_state(self) -> str:
        if self.proto != "TCP":
            return "CONN"
        if self.tcp_rst > 0:
            return "RSTO"
        if self.tcp_fin > 0 and self.tcp_ack > 0:
            return "SF"
        if self.tcp_syn > 0 and self.tcp_ack == 0:
            return "S0"
        if self.tcp_syn > 0 and self.tcp_ack > 0:
            return "S1"
        return "OTH"

    def to_dict(self) -> Dict[str, object]:
        return {
            "ts": round(self.ts, 6),
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "proto": self.proto,
            "service": self.service,
            "duration": round(self.duration, 6),
            "src_bytes": self.src_bytes,
            "dst_bytes": self.dst_bytes,
            "conn_state": self.conn_state(),
            "missed_bytes": self.missed_bytes,
            "src_pkts": self.src_pkts,
            "src_ip_bytes": self.src_ip_bytes,
            "dst_pkts": self.dst_pkts,
            "dst_ip_bytes": self.dst_ip_bytes,
            "dns_query": self.dns_query,
            "dns_qclass": self.dns_qclass,
            "dns_qtype": self.dns_qtype,
            "dns_rcode": self.dns_rcode,
            "dns_AA": self.dns_AA,
            "dns_RD": self.dns_RD,
            "dns_RA": self.dns_RA,
            "dns_rejected": self.dns_rejected,
            "ssl_version": self.ssl_version,
            "ssl_cipher": self.ssl_cipher,
            "ssl_resumed": self.ssl_resumed,
            "ssl_established": self.ssl_established,
            "ssl_subject": self.ssl_subject,
            "ssl_issuer": self.ssl_issuer,
            "http_trans_depth": self.http_trans_depth,
            "http_method": self.http_method,
            "http_uri": self.http_uri,
            "http_referrer": self.http_referrer,
            "http_version": self.http_version,
            "http_request_body_len": self.http_request_body_len,
            "http_response_body_len": self.http_response_body_len,
            "http_status_code": self.http_status_code,
            "http_user_agent": self.http_user_agent,
            "http_orig_mime_types": self.http_orig_mime_types,
            "http_resp_mime_types": self.http_resp_mime_types,
            "weird_name": self.weird_name,
            "weird_addl": self.weird_addl,
            "weird_notice": self.weird_notice,
            "label": self.label,
            "type": self.type,
        }


class PathLabeler:
    def label_and_type(self, pcap_path: Path) -> Tuple[str, str]:
        parts = [p.lower() for p in pcap_path.parts]
        if "normal_pcaps" in parts:
            return "normal", "normal"
        for folder_name, attack_name in ATTACK_FOLDER_MAP.items():
            if folder_name in parts:
                return attack_name, "attack"
        return "unknown", "unknown"


class ServiceResolver:
    def service_from_ports(self, src_port: int, dst_port: int, proto: str) -> str:
        if proto not in {"TCP", "UDP"}:
            return "unknown"
        if dst_port in SERVICE_PORT_MAP:
            return SERVICE_PORT_MAP[dst_port]
        if src_port in SERVICE_PORT_MAP:
            return SERVICE_PORT_MAP[src_port]
        return "unknown"


class FeatureParser:
    """Parseo rápido de capas de aplicación a partir de payload crudo."""

    @staticmethod
    def parse_dns(payload: bytes, flow: FlowRecord) -> None:
        if len(payload) < 12:
            return
        try:
            qdcount = struct.unpack("!H", payload[4:6])[0]
            ancount = struct.unpack("!H", payload[6:8])[0]
            if qdcount < 1:
                return
            idx = 12
            labels = []
            while idx < len(payload):
                length = payload[idx]
                if length == 0:
                    idx += 1
                    break
                if length & 0xC0:
                    break
                idx += 1
                labels.append(payload[idx:idx + length].decode(errors="ignore"))
                idx += length
            if labels:
                flow.dns_query = ".".join(labels)
            if idx + 4 <= len(payload):
                flow.dns_qtype = str(struct.unpack("!H", payload[idx:idx + 2])[0])
                flow.dns_qclass = str(struct.unpack("!H", payload[idx + 2:idx + 4])[0])
            flags = struct.unpack("!H", payload[2:4])[0]
            flow.dns_qr = int(bool(flags & 0x8000)) if hasattr(flow, "dns_qr") else 0
            flow.dns_AA = int(bool(flags & 0x0400))
            flow.dns_RD = int(bool(flags & 0x0100))
            flow.dns_RA = int(bool(flags & 0x0080))
            rcode = flags & 0x000F
            flow.dns_rcode = str(rcode)
            flow.dns_rejected = int(rcode != 0)
            if ancount > 0 and not flow.dns_query:
                flow.dns_query = "response"
        except Exception:
            return

    @staticmethod
    def parse_http(payload: bytes, flow: FlowRecord) -> None:
        if not payload:
            return
        head = payload[:4096]
        if head.startswith(HTTP_METHODS):
            lines = head.split(b"\r\n")
            first = lines[0].decode(errors="ignore")
            parts = first.split()
            if len(parts) >= 2:
                flow.http_method = parts[0]
                flow.http_uri = parts[1]
                flow.http_version = parts[2] if len(parts) >= 3 else ""
                flow.http_trans_depth = 1
                flow.http_request_body_len = len(payload)
            for line in lines[1:]:
                low = line.lower()
                if low.startswith(b"user-agent:"):
                    flow.http_user_agent = line.split(b":", 1)[1].strip().decode(errors="ignore")
                elif low.startswith(b"referer:"):
                    flow.http_referrer = line.split(b":", 1)[1].strip().decode(errors="ignore")
                elif low.startswith(b"content-type:"):
                    flow.http_orig_mime_types = line.split(b":", 1)[1].strip().decode(errors="ignore")
            return
        if head.startswith(b"HTTP/"):
            lines = head.split(b"\r\n")
            first = lines[0].decode(errors="ignore")
            parts = first.split()
            if len(parts) >= 2:
                flow.http_version = parts[0]
                flow.http_status_code = parts[1]
                flow.http_trans_depth = 1
                flow.http_response_body_len = len(payload)
            for line in lines[1:]:
                low = line.lower()
                if low.startswith(b"content-type:"):
                    flow.http_resp_mime_types = line.split(b":", 1)[1].strip().decode(errors="ignore")

    @staticmethod
    def parse_tls(payload: bytes, flow: FlowRecord) -> None:
        if len(payload) < 5:
            return
        # TLS Handshake record
        if payload[0] == 0x16:
            flow.ssl_version = f"{payload[1]}.{payload[2]}"
            flow.ssl_cipher = ""
            flow.ssl_resumed = 0
            flow.ssl_established = 1


class PacketClassifier:
    def __init__(self) -> None:
        self.labeler = PathLabeler()
        self.service_resolver = ServiceResolver()
        self.feature_parser = FeatureParser()

    def classify(self, raw_bytes: bytes, ts: float, pcap_path: Path, flows: Dict[str, FlowRecord]) -> None:
        if len(raw_bytes) < 14:
            return

        eth_type = struct.unpack("!H", raw_bytes[12:14])[0]
        if eth_type == ETH_P_IP:
            self._parse_ipv4(raw_bytes, ts, pcap_path, flows, 14)
        elif eth_type == ETH_P_IPV6:
            self._parse_ipv6(raw_bytes, ts, pcap_path, flows, 14)

    def _parse_ipv4(self, raw: bytes, ts: float, pcap_path: Path, flows: Dict[str, FlowRecord], offset: int) -> None:
        if len(raw) < offset + 20:
            return
        ver_ihl = raw[offset]
        ihl = (ver_ihl & 0x0F) * 4
        if ihl < 20 or len(raw) < offset + ihl:
            return
        proto = raw[offset + 9]
        src_ip = ".".join(str(b) for b in raw[offset + 12:offset + 16])
        dst_ip = ".".join(str(b) for b in raw[offset + 16:offset + 20])
        ip_len = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        l4 = offset + ihl
        payload = raw[l4:]
        src_port = 0
        dst_port = 0
        tcp_flags = 0
        service = "unknown"
        proto_name = "IP"
        if proto == IP_PROTO_TCP and len(raw) >= l4 + 20:
            proto_name = "TCP"
            src_port, dst_port = struct.unpack("!HH", raw[l4:l4 + 4])
            data_offset = ((raw[l4 + 12] >> 4) & 0x0F) * 4
            tcp_flags = raw[l4 + 13]
            payload = raw[l4 + data_offset:]
            service = self.service_resolver.service_from_ports(src_port, dst_port, proto_name)
        elif proto == IP_PROTO_UDP and len(raw) >= l4 + 8:
            proto_name = "UDP"
            src_port, dst_port = struct.unpack("!HH", raw[l4:l4 + 4])
            payload = raw[l4 + 8:]
            service = self.service_resolver.service_from_ports(src_port, dst_port, proto_name)
        elif proto == IP_PROTO_ICMP:
            proto_name = "ICMP"

        flow_id, direction = self._canonical_key(src_ip, src_port, dst_ip, dst_port, proto_name)
        label, typ = self.labeler.label_and_type(pcap_path)

        if flow_id not in flows:
            flows[flow_id] = FlowRecord(
                ts=ts,
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                proto=proto_name,
                service=service,
                label=label,
                type=typ,
                start_ts=ts,
                end_ts=ts,
            )

        flow = flows[flow_id]
        flow.update_time(ts)
        flow.update_direction(direction, len(payload), ip_len, tcp_flags)
        self._parse_application(payload, flow, src_port, dst_port, proto_name)

    def _parse_ipv6(self, raw: bytes, ts: float, pcap_path: Path, flows: Dict[str, FlowRecord], offset: int) -> None:
        if len(raw) < offset + 40:
            return
        next_header = raw[offset + 6]
        src_ip = ":".join(f"{raw[offset + i]:02x}{raw[offset + i + 1]:02x}" for i in range(8, 24, 2))
        dst_ip = ":".join(f"{raw[offset + i]:02x}{raw[offset + i + 1]:02x}" for i in range(24, 40, 2))
        l4 = offset + 40
        payload = raw[l4:]
        src_port = 0
        dst_port = 0
        tcp_flags = 0
        proto_name = "IPV6"
        service = "unknown"
        if next_header == IP_PROTO_TCP and len(raw) >= l4 + 20:
            proto_name = "TCP"
            src_port, dst_port = struct.unpack("!HH", raw[l4:l4 + 4])
            data_offset = ((raw[l4 + 12] >> 4) & 0x0F) * 4
            tcp_flags = raw[l4 + 13]
            payload = raw[l4 + data_offset:]
            service = self.service_resolver.service_from_ports(src_port, dst_port, proto_name)
        elif next_header == IP_PROTO_UDP and len(raw) >= l4 + 8:
            proto_name = "UDP"
            src_port, dst_port = struct.unpack("!HH", raw[l4:l4 + 4])
            payload = raw[l4 + 8:]
            service = self.service_resolver.service_from_ports(src_port, dst_port, proto_name)

        flow_id, direction = self._canonical_key(src_ip, src_port, dst_ip, dst_port, proto_name)
        label, typ = self.labeler.label_and_type(pcap_path)
        if flow_id not in flows:
            flows[flow_id] = FlowRecord(
                ts=ts,
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                proto=proto_name,
                service=service,
                label=label,
                type=typ,
                start_ts=ts,
                end_ts=ts,
            )
        flow = flows[flow_id]
        flow.update_time(ts)
        flow.update_direction(direction, len(payload), 40, tcp_flags)

    def _parse_application(self, payload: bytes, flow: FlowRecord, src_port: int, dst_port: int, proto_name: str) -> None:
        if not payload:
            return
        if proto_name == "UDP" and (src_port == 53 or dst_port == 53):
            self.feature_parser.parse_dns(payload, flow)
        elif proto_name == "TCP":
            if src_port in (80, 8080, 8000, 8008, 8888, 443, 8443) or dst_port in (80, 8080, 8000, 8008, 8888, 443, 8443):
                self.feature_parser.parse_http(payload, flow)
            if src_port == 443 or dst_port == 443 or src_port == 8443 or dst_port == 8443:
                self.feature_parser.parse_tls(payload, flow)

    @staticmethod
    def _canonical_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> Tuple[str, str]:
        left = (src_ip, src_port)
        right = (dst_ip, dst_port)
        if left <= right:
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}", "src_to_dst"
        return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto}", "dst_to_src"


class PcapFileProcessor:
    def __init__(self, pcap_path: Path) -> None:
        self.pcap_path = pcap_path
        self.classifier = PacketClassifier()

    def process(self) -> List[Dict[str, object]]:
        flows: Dict[str, FlowRecord] = {}
        try:
            with RawPcapReader(str(self.pcap_path)) as reader:
                for idx, (raw_bytes, meta) in enumerate(reader):
                    ts = self._timestamp_from_meta(meta, idx)
                    self.classifier.classify(raw_bytes, ts, self.pcap_path, flows)
        except Exception as exc:
            print(f"[WARN] No se pudo procesar {self.pcap_path.name}: {exc}")
            return []
        return [flow.to_dict() for flow in flows.values()]

    @staticmethod
    def _timestamp_from_meta(meta, fallback_index: int) -> float:
        # Compatible con PacketMetadata (pcap) y PacketMetadataNg (pcapng)
        for a, b in (("sec", "usec"), ("tshigh", "tslow"), ("timestamp", None)):
            if hasattr(meta, a):
                if b and hasattr(meta, b):
                    if a == "sec":
                        return float(getattr(meta, "sec")) + float(getattr(meta, "usec")) / 1_000_000.0
                    if a == "tshigh":
                        # Estimación segura para pcapng; suficiente para orden temporal.
                        return float(getattr(meta, "tshigh")) + float(getattr(meta, "tslow")) / 1_000_000_000.0
                value = getattr(meta, a)
                try:
                    return float(value)
                except Exception:
                    pass
        return float(fallback_index)


class DatasetProcessor:
    def __init__(self, root_dir: str | Path, output_name: str = "salida_pcaps.csv") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.output_csv = self.root_dir.parent / output_name

    def discover_pcaps(self) -> List[Path]:
        return sorted(self.root_dir.rglob("*.pcap"))

    def run(self, parallel: bool = True) -> pd.DataFrame:
        pcap_files = self.discover_pcaps()
        if not pcap_files:
            raise FileNotFoundError(f"No se encontraron archivos .pcap dentro de: {self.root_dir}")

        rows: List[Dict[str, object]] = []
        if parallel and len(pcap_files) > 1:
            workers = max(1, (os.cpu_count() or 2) - 1)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_single_pcap, str(p)): p for p in pcap_files}
                for future in as_completed(futures):
                    rows.extend(future.result())
        else:
            for pcap in pcap_files:
                rows.extend(process_single_pcap(str(pcap)))

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            for col in OUTPUT_COLUMNS:
                if col not in df.columns:
                    df[col] = "" if col not in {
                        "ts", "src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "missed_bytes",
                        "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_AA", "dns_RD", "dns_RA",
                        "dns_rejected", "ssl_resumed", "ssl_established", "http_trans_depth",
                        "http_request_body_len", "http_response_body_len"
                    } else 0
            df = df[OUTPUT_COLUMNS]
            df.sort_values(by=["label", "type", "ts"], inplace=True, ignore_index=True)

        df.to_csv(self.output_csv, index=False, encoding="utf-8")
        return df


def process_single_pcap(pcap_path: str) -> List[Dict[str, object]]:
    return PcapFileProcessor(Path(pcap_path)).process()


class App:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_dir = self.base_dir / "Network_dataset_pcaps"
        self.processor = DatasetProcessor(self.dataset_dir)

    def execute(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta esperada: {self.dataset_dir}. Verifica que 'Network_dataset_pcaps' esté al lado de Procesamiento.py."
            )

        df = self.processor.run(parallel=True)
        print("\nProcesamiento finalizado")
        print(f"PCAP raíz: {self.dataset_dir}")
        print(f"CSV generado: {self.processor.output_csv}")
        print(f"Filas generadas: {len(df)}")
        print(f"Columnas: {len(df.columns)}")


def main() -> None:
    App().execute()


if __name__ == "__main__":
    main()
