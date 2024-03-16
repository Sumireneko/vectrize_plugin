from krita import DockWidgetFactory, DockWidgetFactoryBase
from .vectrize import Vectrize

# And add docker:
DOCKER_ID = 'Vectrize'
dock = DockWidgetFactory(DOCKER_ID,DockWidgetFactoryBase.DockRight,Vectrize)
Krita.instance().addDockWidgetFactory(dock)

