from .finetune import FineTuneCL
from .proto_aug import ProtoAugCL
from .proto_evolve import ProtoEvolveCL
from .proto_fedspace import ProtoFedSpaceCL
from .der_pp import DERppCL
from .er_ace import ERAccCL
from .er import ERCL
from .ewc import EWCCL
from .lwf import LwFCL
from .target import TARGETCL
from .gpm import GPMCL
from .fedprotip_vfl import FedProTIPVFLCL
from .prl import PRLCL
from .afc import AFCCL
from .lwf_fim import LwFFIMCL
from .lwf_wa import LwFWACL
from .adagauss import AdaGaussCL
from .proto_evolve_radapt import ProtoEvolveRadaptCL

def get_cl_method(name, trainer, args):
    methods = {'finetune':FineTuneCL, 'proto_aug':ProtoAugCL,
               'proto_evolve':ProtoEvolveCL, 'proto_fedspace':ProtoFedSpaceCL,
               'der_pp':DERppCL, 'er_ace':ERAccCL, 'er':ERCL,
               'ewc':EWCCL, 'lwf':LwFCL, 'target':TARGETCL, 'gpm':GPMCL,
               'fedprotip_vfl':FedProTIPVFLCL,
               'prl':PRLCL, 'afc':AFCCL, 'lwf_fim':LwFFIMCL,
               'lwf_wa':LwFWACL, 'adagauss':AdaGaussCL,
               'proto_evolve_radapt': ProtoEvolveRadaptCL}
    return methods[name](trainer, args)
