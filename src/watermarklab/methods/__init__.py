from __future__ import annotations
from typing import Any

from watermarklab.methods.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from watermarklab.methods.roy2018_dwt_svd import Roy2018DWTSVD
from watermarklab.methods.iwt_hess_svd_2024 import IWTHessSVD2024
from watermarklab.methods.dwt_hd_svd2025 import DWTHDSVD2025
from watermarklab.methods.gaata2022_dwt_hess_fwa import Gaata2022DWTHessFWA
from watermarklab.methods.mahto2022_firefly_dual import Mahto2022FireflyDual
from watermarklab.methods.proposal_qh_dwt_hess import ProposalQHDWTHess, ProposalParams


def build_methods(selected: list[str] | None = None, proposal_options: dict[str, Any] | None = None):
    proposal_options = dict(proposal_options or {})
    params_data = proposal_options.pop("params", None)
    proposal_params = params_data if isinstance(params_data, ProposalParams) else ProposalParams.from_dict(params_data)
    all_methods = {
        "kumar2021": Kumar2021DWTEntropy(),
        "roy2018": Roy2018DWTSVD(),
        "iwt_hess_svd_2024": IWTHessSVD2024(),
        "dwt_hd_svd_2025": DWTHDSVD2025(),
        "gaata2022_dwt_hess_fwa": Gaata2022DWTHessFWA(),
        "mahto2022_firefly_dual": Mahto2022FireflyDual(),
        "proposal": ProposalQHDWTHess(params=proposal_params, **proposal_options),
    }
    if selected is None or selected == ["all"]:
        return all_methods
    return {k: all_methods[k] for k in selected if k in all_methods}


__all__ = [
    "Kumar2021DWTEntropy",
    "Roy2018DWTSVD",
    "IWTHessSVD2024",
    "DWTHDSVD2025",
    "Gaata2022DWTHessFWA",
    "Mahto2022FireflyDual",
    "ProposalQHDWTHess",
    "ProposalParams",
    "build_methods",
]
