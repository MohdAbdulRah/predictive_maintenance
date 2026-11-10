"""Parse an MTConnect /probe document into a flat DataItem catalogue.

Nothing is assumed: IDs, units, categories and the component hierarchy all come
from the document. Namespace-agnostic (works across MTConnect 1.x / 2.x schemas).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from lxml import etree


def _local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


@dataclass
class DataItemDef:
    device_uuid: str
    device_name: str
    id: str
    name: str | None
    category: str            # SAMPLE | EVENT | CONDITION
    type: str
    sub_type: str | None
    units: str | None
    native_units: str | None
    representation: str      # VALUE (default) | TIME_SERIES | DATA_SET | TABLE | DISCRETE
    component_path: str      # e.g. Device/Axes/Linear[X]
    component_type: str
    component_id: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_probe(xml_bytes: bytes) -> dict:
    """Return {'header': {...}, 'devices': [{'uuid','name'}], 'data_items': [DataItemDef,...]}."""
    root = etree.fromstring(xml_bytes)
    if _local(root.tag) == "MTConnectError":
        raise ValueError("Agent returned MTConnectError: " + etree.tostring(root).decode()[:500])

    header = {}
    for el in root:
        if _local(el.tag) == "Header":
            header = {k: v for k, v in el.attrib.items()}

    devices, items = [], []
    for dev_container in root:
        if _local(dev_container.tag) != "Devices":
            continue
        for dev in dev_container:
            if _local(dev.tag) != "Device":
                continue
            uuid = dev.get("uuid") or dev.get("id") or dev.get("name") or "unknown"
            name = dev.get("name") or dev.get("id") or uuid
            devices.append({"uuid": uuid, "name": name, "id": dev.get("id")})
            _walk(dev, uuid, name, name, _local(dev.tag), dev.get("id"), items)
    return {"header": header, "devices": devices, "data_items": items}


def _walk(el, uuid, dname, path, ctype, cid, out):
    for child in el:
        tag = _local(child.tag)
        if tag == "DataItems":
            for di in child:
                if _local(di.tag) != "DataItem":
                    continue
                out.append(DataItemDef(
                    device_uuid=uuid, device_name=dname,
                    id=di.get("id"), name=di.get("name"),
                    category=(di.get("category") or "").upper(),
                    type=(di.get("type") or "").upper(),
                    sub_type=(di.get("subType") or None),
                    units=di.get("units"), native_units=di.get("nativeUnits"),
                    representation=(di.get("representation") or "VALUE").upper(),
                    component_path=path, component_type=ctype, component_id=cid,
                ))
        elif tag == "Components":
            for comp in child:
                label = _local(comp.tag)
                nm = comp.get("name")
                seg = f"{label}[{nm}]" if nm else label
                _walk(comp, uuid, dname, f"{path}/{seg}", label, comp.get("id"), out)
        elif tag in ("Description", "Configuration", "References", "DataItems"):
            continue


def catalogue_summary(parsed: dict) -> list[dict]:
    return [d.to_dict() for d in parsed["data_items"]]
