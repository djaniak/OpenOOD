import os
import urllib.request

from openood.postprocessors import (
    ASHPostprocessor,
    BasePostprocessor,
    CIDERPostprocessor,
    ConfBranchPostprocessor,
    CutPastePostprocessor,
    DICEPostprocessor,
    DRAEMPostprocessor,
    DropoutPostProcessor,
    DSVDDPostprocessor,
    EBOPostprocessor,
    EnsemblePostprocessor,
    GENPostprocessor,
    GMMPostprocessor,
    GodinPostprocessor,
    GradNormPostprocessor,
    GRAMPostprocessor,
    KLMatchingPostprocessor,
    KNNPostprocessor,
    MaxLogitPostprocessor,
    MCDPostprocessor,
    MDSEnsemblePostprocessor,
    MDSPostprocessor,
    MOSPostprocessor,
    NNGuidePostprocessor,
    NPOSPostprocessor,
    ODINPostprocessor,
    OpenGanPostprocessor,
    OpenMax,
    PatchcorePostprocessor,
    RankFeatPostprocessor,
    Rd4adPostprocessor,
    ReactPostprocessor,
    RelationPostprocessor,
    ResidualPostprocessor,
    RMDSPostprocessor,
    RotPredPostprocessor,
    ScalePostprocessor,
    SHEPostprocessor,
    SigmaMeanPostprocessor,
    SigmaStdPostprocessor,
    TemperatureScalingPostprocessor,
    VIMPostprocessor,
)
from openood.postprocessors.bhdsim_postprocessor import BhattacharyyaDistSimPostprocessor
from openood.postprocessors.bhdsimk_postprocessor import BhattacharyyaDistSimKPostprocessor
from openood.postprocessors.edsim_postprocessor import EuclideanDistSimPostprocessor
from openood.postprocessors.edsimk_postprocessor import EuclideanDistSimKPostprocessor
from openood.postprocessors.kldivprior_postprocessor import PriorKlDivPostprocessor
from openood.postprocessors.kldivsim_postprocessor import KlDivSimPostprocessor
from openood.postprocessors.kldivsimk_postprocessor import KlDivSimKPostprocessor
from openood.postprocessors.logpmodel_postprocessor import ModelLogProbPostprocessor
from openood.postprocessors.logpprior_postprocessor import PriorLogProbPostprocessor
from openood.postprocessors.pmds_postprocessor import ProjectedMDSPostprocessor
from openood.postprocessors.pumds_postprocessor import (
    ProjectedUnsupervisedMDSPostprocessor,
)
from openood.postprocessors.rkldivsim_postprocessor import RKlDivSimPostprocessor
from openood.postprocessors.umds_postprocessor import UnsupervisedMDSPostprocessor
from openood.utils.config import Config, merge_configs

postprocessors = {
    "ash": ASHPostprocessor,
    "cider": CIDERPostprocessor,
    "conf_branch": ConfBranchPostprocessor,
    "msp": BasePostprocessor,
    "ebo": EBOPostprocessor,
    "odin": ODINPostprocessor,
    "mds": MDSPostprocessor,
    "mds_ensemble": MDSEnsemblePostprocessor,
    "npos": NPOSPostprocessor,
    "rmds": RMDSPostprocessor,
    "gmm": GMMPostprocessor,
    "patchcore": PatchcorePostprocessor,
    "openmax": OpenMax,
    "react": ReactPostprocessor,
    "vim": VIMPostprocessor,
    "gradnorm": GradNormPostprocessor,
    "godin": GodinPostprocessor,
    "gram": GRAMPostprocessor,
    "cutpaste": CutPastePostprocessor,
    "mls": MaxLogitPostprocessor,
    "residual": ResidualPostprocessor,
    "klm": KLMatchingPostprocessor,
    "temp_scaling": TemperatureScalingPostprocessor,
    "ensemble": EnsemblePostprocessor,
    "dropout": DropoutPostProcessor,
    "draem": DRAEMPostprocessor,
    "dsvdd": DSVDDPostprocessor,
    "mos": MOSPostprocessor,
    "mcd": MCDPostprocessor,
    "opengan": OpenGanPostprocessor,
    "knn": KNNPostprocessor,
    "dice": DICEPostprocessor,
    "scale": ScalePostprocessor,
    "she": SHEPostprocessor,
    "rd4ad": Rd4adPostprocessor,
    "rotpred": RotPredPostprocessor,
    "rankfeat": RankFeatPostprocessor,
    "gen": GENPostprocessor,
    "nnguide": NNGuidePostprocessor,
    "relation": RelationPostprocessor,
    "sigmamean": SigmaMeanPostprocessor,
    "sigmastd": SigmaStdPostprocessor,
    "umds": UnsupervisedMDSPostprocessor,
    "pmds": ProjectedMDSPostprocessor,
    "pumds": ProjectedUnsupervisedMDSPostprocessor,
    "logpprior": PriorLogProbPostprocessor,
    "logpmodel": ModelLogProbPostprocessor,
    "kldivprior": PriorKlDivPostprocessor,
    "kldivsim": KlDivSimPostprocessor,
    "rkldivsim": RKlDivSimPostprocessor,
    "kldivsimk": KlDivSimKPostprocessor,
    "edsim": EuclideanDistSimPostprocessor,
    "edsimk": EuclideanDistSimKPostprocessor,
    "bhdsim": BhattacharyyaDistSimPostprocessor,
    "bhdsimk": BhattacharyyaDistSimKPostprocessor,
}

link_prefix = (
    "https://raw.githubusercontent.com/djaniak/OpenOOD/main/configs/postprocessors/"
)


def get_postprocessor(config_root: str, postprocessor_name: str, id_data_name: str):
    postprocessor_config_path = os.path.join(
        config_root, "postprocessors", f"{postprocessor_name}.yml"
    )
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(
            link_prefix + f"{postprocessor_name}.yml", postprocessor_config_path
        )

    config = Config(postprocessor_config_path)
    config = merge_configs(config, Config(**{"dataset": {"name": id_data_name}}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
