"""ICTC-O2 HP: standalone Cartesian O(2) tensor products."""

__version__ = "0.1.6.dev0"
__all__ = [
    "TensorProduct", "TupleTensorProduct", "ComponentMajorTensorProduct",
    "ReferenceTensorProduct", "UvuTensorProduct", "ReferenceUvuTensorProduct", "__version__",
    "RadialUvuTensorProduct",
    "FlatInference", "compile_inference",
    "DonatedLeftInference",
    "PreparedDonatedLeftInference", "prepare_donated_inference", "build_donated_launcher",
    "GraphInference", "prepare_graph",
]


def __getattr__(name):
    # Importing the package/reference does not require the optional GPU runtime.
    if name in {"PreparedDonatedLeftInference", "prepare_donated_inference", "build_donated_launcher"}:
        from . import donated_deploy
        return getattr(donated_deploy, name)
    if name == "DonatedLeftInference":
        from .donated import DonatedLeftInference
        return DonatedLeftInference
    if name in {"GraphInference", "prepare_graph"}:
        from . import graph
        return getattr(graph,name)
    if name == "ReferenceTensorProduct":
        from .reference import ReferenceTensorProduct
        return ReferenceTensorProduct
    if name == "RadialUvuTensorProduct":
        from .radial_uvu import RadialUvuTensorProduct
        return RadialUvuTensorProduct
    if name in {"UvuTensorProduct", "ReferenceUvuTensorProduct"}:
        from . import uvu_o2
        return getattr(uvu_o2, name)
    if name in {"TensorProduct", "TupleTensorProduct", "ComponentMajorTensorProduct"}:
        from . import api
        return getattr(api, name)
    if name in {"FlatInference", "compile_inference"}:
        from . import aoti
        return getattr(aoti, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
