from dataclasses import dataclass


@dataclass
class FVSpec:
    use_Re_e: bool = False
    use_Re_s: bool = False
    use_Re_m: bool = False
    use_Ve_e: bool = False
    use_Ae_e: bool = False


FVS = {
    "Re": FVSpec(use_Re_e=True),
    "ReVe": FVSpec(use_Re_e=True, use_Ve_e=True),
    "ReVeRs": FVSpec(use_Re_e=True, use_Ve_e=True, use_Re_s=True),
}
