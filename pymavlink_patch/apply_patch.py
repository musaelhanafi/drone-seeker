#!/usr/bin/env python3
"""
Apply TRACKING_MESSAGE (ID 11045) patch to the installed pymavlink dialect.

Run once after installing or upgrading pymavlink:
    python3 pymavlink_patch/apply_patch.py
"""

import importlib
import os
import subprocess
import sys


MARKER = 'name="TRACKING_MESSAGE"'
INSERTION = """\
    <message id="11045" name="TRACKING_MESSAGE">
      <description>Normalised tracking error sent from companion computer to autopilot. errorx and errory are in [-1, 1] where 0 is centred.</description>
      <field type="uint64_t" name="time_usec" units="us">Timestamp (monotonic microseconds)</field>
      <field type="float" name="errorx">Normalised horizontal tracking error [-1, 1]</field>
      <field type="float" name="errory">Normalised vertical tracking error [-1, 1]</field>
    </message>
"""
# Inserted just before the closing tags of the last message block
ANCHOR = "  </messages>\n</mavlink>"


def find_pymavlink_xml() -> str:
    import pymavlink
    base = os.path.dirname(pymavlink.__file__)
    path = os.path.join(base, "dialects", "v20", "ardupilotmega.xml")
    if not os.path.exists(path):
        sys.exit(f"ERROR: cannot find {path}")
    return path


def patch_xml(xml_path: str) -> bool:
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        print("TRACKING_MESSAGE already present in XML — skipping XML patch.")
        return False

    if ANCHOR not in content:
        sys.exit("ERROR: expected anchor not found in ardupilotmega.xml; pymavlink version may be unsupported.")

    patched = content.replace(ANCHOR, INSERTION + ANCHOR)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"Patched {xml_path}")
    return True


def regenerate_dialect(xml_path: str) -> None:
    py_path = xml_path.replace(".xml", ".py")
    from pymavlink.generator import mavgen
    from pymavlink.generator.mavparse import PROTOCOL_2_0

    opts = mavgen.Opts(
        output=py_path,
        wire_protocol=PROTOCOL_2_0,
        language="Python3",
        validate=False,
        strict_units=False,
    )
    mavgen.mavgen(opts, [xml_path])
    print(f"Regenerated {py_path}")


def verify() -> None:
    # Force reimport
    import pymavlink.dialects.v20.ardupilotmega as dialect
    importlib.reload(dialect)
    msg_id = getattr(dialect, "MAVLINK_MSG_ID_TRACKING_MESSAGE", None)
    if msg_id != 11045:
        sys.exit(f"ERROR: verification failed — MAVLINK_MSG_ID_TRACKING_MESSAGE = {msg_id}")
    print(f"Verified: MAVLINK_MSG_ID_TRACKING_MESSAGE = {msg_id}")


if __name__ == "__main__":
    xml_path = find_pymavlink_xml()
    changed = patch_xml(xml_path)
    if changed:
        regenerate_dialect(xml_path)
    verify()
    print("Done.")
