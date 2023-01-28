#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import datetime
from vedo import __version__, printc

cmd =  "pdoc vedo -o html -t . "
cmd+= f'--footer-text "version {__version__}, rev {datetime.date.today()}." '
cmd+=  '--logo https://vedo.embl.es/images/logos/logo_vedo_simple_transp.png '
cmd+=  '--favicon https://vedo.embl.es/images/logos/favicon.svg '
printc("Generating documentation:\n", cmd, "\n..please wait", c='y')
os.system(cmd)
os.system("chmod 755 html/ -R")
printc("Done.", c='y')

printc("Move to server manually with commands:")
printc(" ls ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/")
printc(" rm ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/html")
printc(" mv html/ ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/")
