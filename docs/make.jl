using PiccoloQuantumObjects
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Manual" => [
        "Isomorphisms" => "generated/isomorphisms.md",
        "Quantum Objects" => "generated/quantum_objects.md",
        "Quantum Systems" => "generated/quantum_systems.md",
        "Rollouts" => "generated/rollouts.md",
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