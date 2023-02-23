import flower_classifier.metadata.shared as shared


RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "flowers"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "flowers"
DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME

DIMS = (3, 224, 224)
OUTPUT_DIMS = (1,)
MAPPING = [
    "fire lily",
    "canterbury bells",
    "bolero deep blue",
    "pink primrose",
    "mexican aster",
    "prince of wales feathers",
    "moon orchid",
    "globe-flower",
    "grape hyacinth",
    "corn poppy",
    "toad lily",
    "siam tulip",
    "red ginger",
    "spring crocus",
    "alpine sea holly",
    "garden phlox",
    "globe thistle",
    "tiger lily",
    "ball moss",
    "love in the mist",
    "monkshood",
    "blackberry lily",
    "spear thistle",
    "balloon flower",
    "blanket flower",
    "king protea",
    "oxeye daisy",
    "yellow iris",
    "cautleya spicata",
    "carnation",
    "silverbush",
    "bearded iris",
    "black-eyed susan",
    "windflower",
    "japanese anemone",
    "giant white arum lily",
    "great masterwort",
    "sweet pea",
    "tree mallow",
    "trumpet creeper",
    "daffodil",
    "pincushion flower",
    "hard-leaved pocket orchid",
    "sunflower",
    "osteospermum",
    "tree poppy",
    "desert-rose",
    "bromelia",
    "magnolia",
    "english marigold",
    "bee balm",
    "stemless gentian",
    "mallow",
    "gaura",
    "lenten rose",
    "marigold",
    "orange dahlia",
    "buttercup",
    "pelargonium",
    "ruby-lipped cattleya",
    "hippeastrum",
    "artichoke",
    "gazania",
    "canna lily",
    "peruvian lily",
    "mexican petunia",
    "bird of paradise",
    "sweet william",
    "purple coneflower",
    "wild pansy",
    "columbine",
    "colt's foot",
    "snapdragon",
    "camellia",
    "fritillary",
    "common dandelion",
    "poinsettia",
    "primula",
    "azalea",
    "californian poppy",
    "anthurium",
    "morning glory",
    "cape flower",
    "bishop of llandaff",
    "pink-yellow dahlia",
    "clematis",
    "geranium",
    "thorn apple",
    "barbeton daisy",
    "bougainvillea",
    "sword lily",
    "hibiscus",
    "lotus lotus",
    "cyclamen",
    "foxglove",
    "frangipani",
    "rose",
    "watercress",
    "water lily",
    "wallflower",
    "passion flower",
    "petunia",
]
