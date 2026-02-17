#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import subprocess
from pathlib import Path

from vedo import __version__, printc

script_dir = Path(__file__).resolve().parent
output_dir = Path(os.getenv("VEDO_PDOC_OUTPUT", str(script_dir / "html"))).expanduser().resolve()
cmd = [
    "pdoc",
    "vedo",
    "-o",
    str(output_dir),
    "-t",
    str(script_dir),
    "--footer-text",
    f"version {__version__}, rev {datetime.date.today()}.",
    "--logo",
    "https://vedo.embl.es/images/logos/logo_vedo_simple_transp.png",
    "--favicon",
    "https://vedo.embl.es/images/logos/favicon.svg",
]
printc("Generating documentation:\n", " ".join(cmd), "\n..please wait", c="y")
subprocess.run(cmd, check=True)
for root, dirs, files in os.walk(output_dir):
    root_path = Path(root)
    os.chmod(root_path, 0o755)
    for name in dirs:
        os.chmod(root_path / name, 0o755)
    for name in files:
        os.chmod(root_path / name, 0o644)
printc("Done.", c="y")

printc("Move to server manually with commands:")
printc(" ls ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/")
printc(" rm ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/html")
printc(f" mv {output_dir}/ ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/")
