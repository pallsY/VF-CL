from .retrain import RetrainUL
from .gradient_ascent import GradientAscentUL
from .luv import LUVUL
from .mode import MoDeUL
from .fucrt import FUCRTUL
from .fedup import FedUPUL
from .fedosd import FedOSDUL
from .fedau import FedAUUL
from .fudp import FUDPUL
from .radapt_router import RadaptRouterUL
from .roar import RoarUL


def get_ul_method(name, trainer, args):
    return {
        'retrain':         RetrainUL,
        'gradient_ascent': GradientAscentUL,
        'luv':             LUVUL,
        'mode':            MoDeUL,
        'fucrt':           FUCRTUL,
        'fedup':           FedUPUL,
        'fedosd':          FedOSDUL,
        'fedau':           FedAUUL,
        'fudp':            FUDPUL,
        'radapt_router':   RadaptRouterUL,
        'roar':            RoarUL,
    }[name](trainer, args)
