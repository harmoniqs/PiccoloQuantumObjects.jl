using PiccoloQuantumObjects
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Quickstart" => "generated/quickstart.md",
    "Manual" => [
        "Quantum Systems" => "generated/quantum_systems.md",
        "Quantum Objects" => "generated/quantum_objects.md",
        "Rollouts" => "generated/rollouts.md",
        "Isomorphisms" => "generated/isomorphisms.md",
    ],
    "Library" => "lib.md",
]

generate_docs(
    @__DIR__,
    "PiccoloQuantumObjects",
    PiccoloQuantumObjects,
    pages;
    format_kwargs = (canonical = "https://docs.harmoniqs.co/PiccoloQuantumObjects.jl",),
)